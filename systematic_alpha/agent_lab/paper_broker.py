from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import requests

from systematic_alpha.agent_lab.storage import AgentLabStorage
from systematic_alpha.credentials import load_credentials
from systematic_alpha.mojito_loader import import_mojito_module


class PaperBroker:
    def __init__(self, storage: AgentLabStorage):
        self.storage = storage
        self.execution_mode = str(os.getenv("AGENT_LAB_EXECUTION_MODE", "mojito_mock")).strip().lower()
        self.us_exchange = str(os.getenv("AGENT_LAB_US_EXCHANGE", "NASD")).strip().upper() or "NASD"
        self._brokers: Dict[str, Any] = {}
        self._creds_loaded = False
        self._creds: Dict[str, Any] = {}

    def _load_creds(self) -> None:
        if self._creds_loaded:
            return
        self._creds_loaded = True
        try:
            key, secret, acc_no, _ = load_credentials(None)
            self._creds = {"key": key, "secret": secret, "acc_no": acc_no}
        except Exception:
            self._creds = {}

    @staticmethod
    def _normalize_us_exchange(raw: str) -> str:
        upper = str(raw or "").strip().upper()
        return {
            "NASD": "NASD",
            "NASDAQ": "NASD",
            "NYSE": "NYSE",
            "NYS": "NYSE",
            "AMEX": "AMEX",
            "AMS": "AMEX",
        }.get(upper, upper or "NASD")

    def _exchange_candidates(self, market: str) -> List[str]:
        if market.upper() == "KR":
            return ["KR"]
        ordered = [self._normalize_us_exchange(self.us_exchange), "NASD", "NYSE", "AMEX"]
        out: List[str] = []
        seen = set()
        for ex in ordered:
            if ex in seen:
                continue
            seen.add(ex)
            out.append(ex)
        return out

    def _exchange_label(self, market: str, us_exchange_code: str = "") -> str:
        # self-heal:us-exchange-resolver-v1
        if market.upper() == "KR":
            return "KR"

        normalized = self._normalize_us_exchange(us_exchange_code or self.us_exchange)

        try:
            mojito = import_mojito_module()
            ki = importlib.import_module(mojito.__name__ + ".koreainvestment")
            exchange_code3 = getattr(ki, "EXCHANGE_CODE3", {})
            if isinstance(exchange_code3, dict):
                for label, code in exchange_code3.items():
                    if str(code).upper() == normalized:
                        return str(label)
        except Exception:
            pass

        fallback = {
            "NASD": "나스닥",
            "NYSE": "뉴욕",
            "AMEX": "아멕스",
        }
        return fallback.get(normalized, "나스닥")

    @staticmethod
    def _session_date_for_market(market: str) -> str:
        market_upper = str(market or "").strip().upper()
        tz_name = "Asia/Seoul" if market_upper == "KR" else "America/New_York"
        try:
            return datetime.now(ZoneInfo(tz_name)).strftime("%Y%m%d")
        except Exception:
            return datetime.now().strftime("%Y%m%d")

    @staticmethod
    def _daily_untradable_meta_key(market: str, session_date: str) -> str:
        return f"daily_untradable_symbols:{str(market or '').strip().upper()}:{str(session_date or '').strip()}"

    def _load_daily_untradable_symbols(self, market: str, session_date: str) -> set[str]:
        key = self._daily_untradable_meta_key(market, session_date)
        raw = self.storage.get_system_meta(key, "[]")
        out: set[str] = set()
        try:
            data = json.loads(str(raw or "[]"))
        except Exception:
            data = []
        if isinstance(data, list):
            for item in data:
                sym = str(item or "").strip().upper()
                if sym:
                    out.add(sym)
        return out

    def _save_daily_untradable_symbols(self, market: str, session_date: str, symbols: set[str]) -> None:
        key = self._daily_untradable_meta_key(market, session_date)
        payload = sorted(str(x).strip().upper() for x in symbols if str(x).strip())
        self.storage.upsert_system_meta(
            key,
            json.dumps(payload, ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        )

    def _append_daily_untradable_symbol(self, market: str, session_date: str, symbol: str) -> bool:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return False
        current = self._load_daily_untradable_symbols(market, session_date)
        if sym in current:
            return False
        current.add(sym)
        self._save_daily_untradable_symbols(market, session_date, current)
        return True

    @staticmethod
    def _is_non_tradable_error_text(text: str) -> bool:
        raw = str(text or "")
        norm = raw.lower()
        return ("msg_cd=40070000" in norm) or ("매매불가 종목" in raw)

    @staticmethod
    def _collect_send_error_text(send_result: Dict[str, Any]) -> str:
        if not isinstance(send_result, dict):
            return ""
        chunks: List[str] = []
        reason = send_result.get("reason")
        if reason:
            chunks.append(str(reason))
        attempts = send_result.get("attempts")
        if isinstance(attempts, list):
            for row in attempts:
                if not isinstance(row, dict):
                    continue
                rr = row.get("reason")
                if rr:
                    chunks.append(str(rr))
        return " | ".join(chunks)

    @classmethod
    def _extract_reject_reason(cls, send_result: Dict[str, Any]) -> str:
        if not isinstance(send_result, dict):
            return ""
        reason = str(send_result.get("reason", "") or "").strip()
        attempts = send_result.get("attempts")
        if reason == "all_exchange_attempts_failed" and isinstance(attempts, list):
            attempt_reasons: List[str] = []
            for row in attempts:
                if not isinstance(row, dict):
                    continue
                rr = str(row.get("reason", "") or "").strip()
                if rr:
                    attempt_reasons.append(rr)
            if attempt_reasons:
                reason = "; ".join(attempt_reasons[:2])
        if not reason:
            reason = cls._collect_send_error_text(send_result)
        return str(reason or "").strip()

    def _has_recent_sell_submission(self, market: str, symbol: str, lookback_sec: int) -> bool:
        if int(lookback_sec) <= 0:
            return False
        include_filled = self._env_bool("AGENT_LAB_SELL_REPEAT_GUARD_INCLUDE_FILLED", False)
        statuses = ["SUBMITTED"]
        if include_filled:
            statuses.append("FILLED")
        placeholders = ",".join(["?"] * len(statuses))
        rows = self.storage.query_all(
            f"""
            SELECT side, status, submitted_at
            FROM paper_orders
            WHERE market = ? AND symbol = ? AND status IN ({placeholders})
            ORDER BY submitted_at DESC
            LIMIT 1
            """,
            (
                str(market or "").strip().upper(),
                str(symbol or "").strip().upper(),
                *statuses,
            ),
        )
        if not rows:
            return False
        row = rows[0]
        if str(row.get("side", "")).strip().upper() != "SELL":
            return False
        submitted_at = str(row.get("submitted_at", "")).strip()
        try:
            last_dt = datetime.fromisoformat(submitted_at)
        except Exception:
            return False
        elapsed = (datetime.now() - last_dt).total_seconds()
        return float(elapsed) <= float(lookback_sec)

    def _get_broker(self, market: str, us_exchange_code: str = ""):
        market_upper = market.upper()
        exchange_code = self._normalize_us_exchange(us_exchange_code or self.us_exchange)
        key = f"{market_upper}:{exchange_code}" if market_upper == "US" else market_upper
        if key in self._brokers:
            return self._brokers[key]
        self._load_creds()
        if not self._creds:
            return None
        try:
            mojito = import_mojito_module()
            if key == "KR":
                # Domestic mock account uses default exchange routing.
                broker = mojito.KoreaInvestment(
                    api_key=self._creds["key"],
                    api_secret=self._creds["secret"],
                    acc_no=self._creds["acc_no"],
                    mock=True,
                )
            else:
                broker = mojito.KoreaInvestment(
                    api_key=self._creds["key"],
                    api_secret=self._creds["secret"],
                    acc_no=self._creds["acc_no"],
                    exchange=self._exchange_label(market_upper, exchange_code),
                    mock=True,
                )
            self._brokers[key] = broker
            return broker
        except Exception:
            return None

    def _send_order(self, market: str, order: Dict[str, Any]) -> Dict[str, Any]:
        market_upper = str(market or "").strip().upper()
        symbol = str(order["symbol"])
        qty = int(float(order["quantity"]))
        order_type = str(order.get("order_type", "MARKET")).upper()
        side = str(order.get("side", "BUY")).upper()
        effective_order_type = str(order_type)
        effective_limit_price: float | None = None
        if market_upper == "US" and order_type == "MARKET" and self._env_bool("AGENT_LAB_US_MARKET_AS_AGGRESSIVE_LIMIT", True):
            ref_price = self._to_float(order.get("reference_price"), 0.0)
            aggressive_px = self._compute_us_aggressive_limit(side=side, ref_price=ref_price)
            if aggressive_px is not None and aggressive_px > 0:
                effective_order_type = "LIMIT"
                effective_limit_price = aggressive_px
        attempts: List[Dict[str, Any]] = []
        for exchange_code in self._exchange_candidates(market):
            broker = self._get_broker(market, exchange_code)
            if broker is None:
                attempts.append({"exchange": exchange_code, "ok": False, "reason": "broker_unavailable"})
                continue
            try:
                if effective_order_type == "LIMIT":
                    raw_px = effective_limit_price
                    if raw_px is None:
                        raw_px = self._to_float(order.get("limit_price") or order.get("reference_price"), 0.0)
                    if market_upper == "US":
                        px = float(max(0.0001, round(float(raw_px), 4)))
                    else:
                        px = int(max(1, round(float(raw_px))))
                    if side == "BUY":
                        resp = self._call_with_rate_limit_retry(
                            lambda b=broker, s=symbol, p=px, q=qty: b.create_limit_buy_order(s, p, q),
                            op_name=f"{market}.{exchange_code}.limit_buy.{symbol}",
                        )
                    else:
                        resp = self._call_with_rate_limit_retry(
                            lambda b=broker, s=symbol, p=px, q=qty: b.create_limit_sell_order(s, p, q),
                            op_name=f"{market}.{exchange_code}.limit_sell.{symbol}",
                        )
                else:
                    if side == "BUY":
                        resp = self._call_with_rate_limit_retry(
                            lambda b=broker, s=symbol, q=qty: b.create_market_buy_order(s, q),
                            op_name=f"{market}.{exchange_code}.market_buy.{symbol}",
                        )
                    else:
                        resp = self._call_with_rate_limit_retry(
                            lambda b=broker, s=symbol, q=qty: b.create_market_sell_order(s, q),
                            op_name=f"{market}.{exchange_code}.market_sell.{symbol}",
                        )
                api_err = self._api_error_text(resp if isinstance(resp, dict) else {})
                if api_err:
                    attempts.append({"exchange": exchange_code, "ok": False, "reason": api_err})
                    continue
                return {
                    "ok": True,
                    "response": resp,
                    "exchange": exchange_code,
                    "attempts": attempts,
                    "effective_order_type": effective_order_type,
                    "effective_limit_price": effective_limit_price,
                }
            except Exception as exc:
                attempts.append({"exchange": exchange_code, "ok": False, "reason": repr(exc)})
                continue
        return {
            "ok": False,
            "reason": "all_exchange_attempts_failed",
            "attempts": attempts,
            "effective_order_type": effective_order_type,
            "effective_limit_price": effective_limit_price,
        }

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            text = str(value).strip().replace(",", "")
            if text == "":
                return float(default)
            return float(text)
        except Exception:
            return float(default)

    @classmethod
    def _pick_value(cls, row: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in row:
                value = cls._to_float(row.get(key), default=default)
                if value != default:
                    return value
        return cls._to_float(row.get(keys[0], default), default=default) if keys else float(default)

    @classmethod
    def _pick_present_value(cls, row: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
        """
        Return the first *present* field value, even if it is 0.
        This avoids treating valid zero-values as "missing".
        """
        for key in keys:
            if key not in row:
                continue
            raw = row.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text == "":
                continue
            return cls._to_float(raw, default=default)
        return float(default)

    @classmethod
    def _parse_balance_domestic(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        output1 = payload.get("output1", [])
        output2 = payload.get("output2", [])
        if not isinstance(output1, list):
            output1 = []
        if not isinstance(output2, list):
            output2 = []
        summary = output2[0] if output2 else {}
        if not isinstance(summary, dict):
            summary = {}

        cash_krw = cls._pick_present_value(
            summary,
            ["ord_psbl_cash", "wdrw_psbl_amt", "wdrw_psbl_tot_amt", "dnca_tot_amt", "dnca_tot_amt2", "tot_dnca"],
            default=0.0,
        )
        equity_krw = cls._pick_present_value(
            summary,
            ["tot_evlu_amt", "nass_amt", "tot_asst_amt", "tot_eval_amt", "scts_evlu_amt"],
            default=cash_krw,
        )
        positions: List[Dict[str, Any]] = []
        for row in output1:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("pdno") or row.get("symbol") or "").strip().upper()
            qty = cls._pick_value(row, ["hldg_qty", "hold_qty", "ord_psbl_qty", "qty"])
            if not symbol or qty <= 0:
                continue
            avg_price = cls._pick_value(row, ["pchs_avg_pric", "pchs_pric", "avg_pric", "avg_price"])
            market_value = cls._pick_value(row, ["evlu_amt", "evlu_pfls_amt", "evlu_erng_rt", "market_value"])
            if market_value <= 0 and avg_price > 0:
                market_value = avg_price * qty
            positions.append(
                {
                    "market": "KR",
                    "symbol": symbol,
                    "quantity": float(qty),
                    "avg_price": float(avg_price),
                    "market_value_krw": float(max(0.0, market_value)),
                    "currency": "KRW",
                    "fx_rate": 1.0,
                    "payload": row,
                }
            )
        positions_value_krw = float(sum(float(p.get("market_value_krw", 0.0) or 0.0) for p in positions))
        if float(cash_krw) < 0.0:
            # Domestic mock/live payload can expose dnca_tot_amt as negative (settlement-centric),
            # which is not suitable as trading cash. Prefer non-negative proxy fields first.
            proxy_cash = cls._pick_present_value(
                summary,
                ["ord_psbl_cash", "wdrw_psbl_amt", "wdrw_psbl_tot_amt", "prvs_rcdl_excc_amt"],
                default=-1.0,
            )
            if float(proxy_cash) >= 0.0:
                cash_krw = float(proxy_cash)
            elif float(equity_krw) > 0.0 and positions_value_krw >= 0.0:
                derived = float(equity_krw) - positions_value_krw
                if derived >= 0.0:
                    cash_krw = float(derived)
        # Keep account figures API-authoritative.
        # If there are no holdings and equity is positive, prefer total equity as cash proxy.
        if len(positions) == 0 and float(equity_krw) > 0.0 and float(cash_krw) <= 0.0:
            cash_krw = float(equity_krw)
        return {
            "market": "KR",
            "cash_krw": float(cash_krw),
            "equity_krw": float(max(equity_krw, cash_krw)),
            "positions": positions,
            "raw": payload,
        }

    @classmethod
    def _parse_balance_oversea(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        output1, output2 = cls._normalize_oversea_payload(payload)
        if not isinstance(output1, list):
            output1 = []
        if not isinstance(output2, list):
            output2 = []
        summary_rows = [row for row in output2 if isinstance(row, dict)]

        # Overseas output2 is often multi-currency rows; first row can be non-USD zeros.
        cash_candidates: List[float] = []
        equity_candidates: List[float] = []
        for summary in summary_rows:
            cash_candidates.append(
                cls._pick_value(
                    summary,
                    [
                        "frcr_drwg_psbl_amt_1",
                        "nxdy_frcr_drwg_psbl_amt",
                        "ovrs_ord_psbl_amt",
                        "cash_amt",
                        "frcr_evlu_pfls_amt",
                        "tot_evlu_pfls_amt",
                    ],
                    default=0.0,
                )
            )
            equity_candidates.append(
                cls._pick_value(
                    summary,
                    ["tot_evlu_amt", "frcr_buy_amt_smtl", "ovrs_tot_pfls", "tot_asst_amt", "frcr_evlu_amt2"],
                    default=0.0,
                )
            )

        cash_krw = max(cash_candidates) if cash_candidates else 0.0
        equity_krw = max(equity_candidates) if equity_candidates else cash_krw
        positions: List[Dict[str, Any]] = []
        for row in output1:
            if not isinstance(row, dict):
                continue
            symbol = str(
                row.get("ovrs_pdno")
                or row.get("pdno")
                or row.get("symb")
                or row.get("symbol")
                or ""
            ).strip().upper()
            qty = cls._pick_value(
                row,
                [
                    "cblc_qty13",
                    "ovrs_cblc_qty",
                    "hldg_qty",
                    "hold_qty",
                    "ord_psbl_qty1",
                    "ord_psbl_qty",
                    "qty",
                ],
            )
            if not symbol or qty <= 0:
                continue
            avg_price = cls._pick_value(row, ["avg_unpr3", "pchs_avg_pric", "avg_pric", "avg_price"])
            fx_rate = cls._pick_value(row, ["bass_exrt", "frcr_exrt", "fx_rate"], default=1.0)
            if fx_rate <= 0:
                fx_rate = 1.0
            market_value_krw = cls._pick_value(
                row,
                ["frcr_evlu_amt2", "evlu_pfls_amt", "evlu_amt", "market_value_krw"],
            )
            if market_value_krw <= 0:
                local_px = cls._pick_value(row, ["ovrs_now_pric1", "now_pric2", "last", "close"])
                if local_px <= 0:
                    local_px = avg_price
                market_value_krw = qty * local_px * fx_rate
            positions.append(
                {
                    "market": "US",
                    "symbol": symbol,
                    "quantity": float(qty),
                    "avg_price": float(avg_price * fx_rate),
                    "market_value_krw": float(max(0.0, market_value_krw)),
                    "currency": str(row.get("tr_crcy_cd") or row.get("crcy_cd") or "USD").upper(),
                    "fx_rate": float(fx_rate),
                    "payload": row,
                }
            )
        positions_value_krw = float(sum(float(p.get("market_value_krw", 0.0) or 0.0) for p in positions))
        if float(equity_krw) <= 0.0 and positions_value_krw > 0.0:
            equity_krw = float(positions_value_krw)
        if float(cash_krw) < 0.0:
            cash_krw = 0.0
        if float(cash_krw) <= 0.0 and float(equity_krw) > 0.0 and positions_value_krw >= 0.0:
            derived = float(equity_krw) - positions_value_krw
            if derived >= 0.0:
                cash_krw = float(derived)
        return {
            "market": "US",
            "cash_krw": float(cash_krw),
            "equity_krw": float(max(equity_krw, cash_krw)),
            "positions": positions,
            "raw": payload,
        }

    @staticmethod
    def _pick_text_ci(row: Dict[str, Any], keys: List[str]) -> str:
        if not isinstance(row, dict):
            return ""
        for key in keys:
            if key in row:
                value = row.get(key)
                text = str(value or "").strip()
                if text:
                    return text
        lowered = {str(k).strip().lower(): v for k, v in row.items()}
        for key in keys:
            value = lowered.get(str(key).strip().lower())
            text = str(value or "").strip()
            if text:
                return text
        return ""

    @classmethod
    def _normalize_side_text(cls, raw: str) -> str:
        text = str(raw or "").strip().upper()
        if not text:
            return ""
        if text in {"1", "01", "02_BUY", "BUY", "B"}:
            return "BUY"
        if text in {"2", "02", "01_SELL", "SELL", "S"}:
            return "SELL"
        if "매수" in raw:
            return "BUY"
        if "매도" in raw:
            return "SELL"
        return text

    @classmethod
    def _parse_open_order_rows(
        cls,
        *,
        market: str,
        exchange: str,
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        for key in ("output", "output1", "output2"):
            raw = payload.get(key)
            if isinstance(raw, dict):
                rows.append(raw)
            elif isinstance(raw, list):
                rows.extend([row for row in raw if isinstance(row, dict)])
        seen: Set[str] = set()
        market_upper = str(market or "").strip().upper()
        ex_upper = str(exchange or "").strip().upper()
        for row in rows:
            broker_order_id = cls._pick_text_ci(
                row,
                [
                    "ODNO",
                    "odno",
                    "ord_no",
                    "ordno",
                    "ORGN_ODNO",
                    "orgn_odno",
                    "ovrs_ord_no",
                    "OVRS_ORD_NO",
                    "ord_sqn",
                    "ORD_SQN",
                ],
            ).strip()
            if not broker_order_id:
                continue
            unique_key = f"{market_upper}:{broker_order_id}"
            if unique_key in seen:
                continue
            seen.add(unique_key)
            symbol = cls._pick_text_ci(row, ["PDNO", "pdno", "ovrs_pdno", "symb", "symbol"]).strip().upper()
            side = cls._normalize_side_text(
                cls._pick_text_ci(
                    row,
                    [
                        "SLL_BUY_DVSN_CD",
                        "sll_buy_dvsn_cd",
                        "sll_buy_dvsn_name",
                        "SLL_BUY_DVSN_NAME",
                        "side",
                        "ord_dvsn",
                        "ORD_DVSN",
                    ],
                )
            )
            qty = cls._pick_value(
                row,
                [
                    "ORD_QTY",
                    "ord_qty",
                    "RVSE_CNCL_QTY",
                    "rvse_cncl_qty",
                    "ord_psbl_qty",
                    "ord_psbl_qty1",
                    "nccs_qty",
                    "NCCS_QTY",
                    "ft_ord_qty",
                    "FT_ORD_QTY",
                    "qty",
                ],
                default=0.0,
            )
            out.append(
                {
                    "market": market_upper,
                    "exchange": ex_upper,
                    "broker_order_id": broker_order_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": float(max(0.0, qty)),
                    "raw": row,
                }
            )
        return out

    @staticmethod
    def _env_csv(name: str, default_values: List[str]) -> List[str]:
        raw = str(os.getenv(name, "") or "").strip()
        if not raw:
            return list(default_values)
        out: List[str] = []
        seen: set[str] = set()
        for token in raw.split(","):
            item = str(token or "").strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out or list(default_values)

    def _fetch_open_orders_us_direct_exchange(self, exchange_code: str) -> Dict[str, Any]:
        exchange_upper = str(exchange_code or "").strip().upper()
        broker = self._get_broker("US", exchange_upper)
        if broker is None:
            return {
                "ok": False,
                "open_orders": [],
                "errors": [f"{exchange_upper}:broker_unavailable"],
                "best_effort": True,
            }

        base_url = str(getattr(broker, "base_url", "") or "").strip()
        app_key = str(getattr(broker, "api_key", "") or "").strip()
        app_secret = str(getattr(broker, "api_secret", "") or "").strip()
        cano = str(getattr(broker, "acc_no_prefix", "") or "").strip()
        acnt_prdt_cd = str(getattr(broker, "acc_no_postfix", "") or "").strip()
        is_mock = bool(getattr(broker, "mock", True))
        if not (base_url and app_key and app_secret and cano and acnt_prdt_cd):
            return {
                "ok": False,
                "open_orders": [],
                "errors": [f"{exchange_upper}:broker_fields_missing"],
                "best_effort": True,
            }

        timeout_sec = self._env_float("AGENT_LAB_US_OPEN_ORDER_TIMEOUT_SEC", 8.0, 1.0, 60.0)
        max_pages = self._env_int("AGENT_LAB_US_OPEN_ORDER_MAX_PAGES", 1, 1, 10)
        default_paths = ["/uapi/overseas-stock/v1/trading/inquire-nccs"]
        default_tr_ids = ["VTTS3018R", "VTTT3018R"] if is_mock else ["TTTS3018R", "JTTT3018R"]
        paths = self._env_csv("AGENT_LAB_US_OPEN_ORDER_PATHS", default_paths)
        tr_ids = self._env_csv(
            "AGENT_LAB_US_OPEN_ORDER_TR_IDS_MOCK" if is_mock else "AGENT_LAB_US_OPEN_ORDER_TR_IDS_REAL",
            default_tr_ids,
        )

        out: Dict[str, Any] = {
            "ok": False,
            "open_orders": [],
            "errors": [],
            "attempts": [],
            "best_effort": True,
        }
        seen_order_ids: set[str] = set()
        for path in paths:
            norm_path = str(path or "").strip()
            if not norm_path:
                continue
            if not norm_path.startswith("/"):
                norm_path = "/" + norm_path
            url = f"{base_url}{norm_path}"
            for tr_id in tr_ids:
                tr = str(tr_id or "").strip()
                if not tr:
                    continue
                ctx_fk200 = ""
                ctx_nk200 = ""
                got_success = False
                for page in range(1, max_pages + 1):
                    params = {
                        "CANO": cano,
                        "ACNT_PRDT_CD": acnt_prdt_cd,
                        "OVRS_EXCG_CD": exchange_upper,
                        "SORT_SQN": "DS",
                        "CTX_AREA_FK200": ctx_fk200,
                        "CTX_AREA_NK200": ctx_nk200,
                    }
                    try:
                        def _request_page(
                            *,
                            u: str = url,
                            p: Dict[str, Any] = dict(params),
                            tr_id: str = tr,
                            b: Any = broker,
                            key: str = app_key,
                            secret: str = app_secret,
                        ) -> Dict[str, Any]:
                            auth = str(getattr(b, "access_token", "") or "").strip()
                            headers = {
                                "content-type": "application/json",
                                "authorization": auth,
                                "appKey": key,
                                "appSecret": secret,
                                "tr_id": tr_id,
                            }
                            return self._http_get_json(
                                url=u,
                                headers=headers,
                                params=p,
                                timeout_sec=timeout_sec,
                            )

                        payload = self._call_with_rate_limit_retry(
                            _request_page,
                            op_name=f"US.{exchange_upper}.open_orders.{tr}.page{page}",
                        )
                    except Exception as exc:
                        out["errors"].append(f"{exchange_upper}:{norm_path}:{tr}:{repr(exc)}")
                        break
                    if not isinstance(payload, dict):
                        out["errors"].append(f"{exchange_upper}:{norm_path}:{tr}:invalid_payload_type")
                        break
                    api_err = self._api_error_text(payload)
                    if api_err:
                        out["errors"].append(f"{exchange_upper}:{norm_path}:{tr}:{api_err}")
                        break

                    got_success = True
                    out["attempts"].append(
                        {
                            "exchange": exchange_upper,
                            "path": norm_path,
                            "tr_id": tr,
                            "page": int(page),
                            "msg_cd": str(payload.get("msg_cd", "") or ""),
                            "msg1": str(payload.get("msg1", "") or ""),
                        }
                    )
                    rows = self._parse_open_order_rows(market="US", exchange=exchange_upper, payload=payload)
                    for row in rows:
                        odno = str(row.get("broker_order_id", "")).strip()
                        if not odno or odno in seen_order_ids:
                            continue
                        seen_order_ids.add(odno)
                        out["open_orders"].append(row)

                    tr_cont = str(payload.get("tr_cont", "") or "").strip().upper()
                    next_fk = str(payload.get("ctx_area_fk200", "") or "").strip()
                    next_nk = str(payload.get("ctx_area_nk200", "") or "").strip()
                    if tr_cont in {"", "E"} or (not next_fk and not next_nk):
                        break
                    ctx_fk200 = next_fk
                    ctx_nk200 = next_nk
                    time.sleep(self._env_float("AGENT_LAB_US_OPEN_ORDER_PAGE_DELAY_SEC", 0.2, 0.0, 2.0))
                if got_success:
                    out["ok"] = True
        return out

    @staticmethod
    def _http_get_json(
        *,
        url: str,
        headers: Dict[str, Any],
        params: Dict[str, Any],
        timeout_sec: float,
    ) -> Dict[str, Any]:
        resp = requests.get(url, headers=headers, params=params, timeout=float(timeout_sec))
        try:
            payload = resp.json()
        except Exception:
            payload = {"rt_cd": "1", "msg_cd": "HTTP_JSON_PARSE_ERROR", "msg1": str(resp.text[:200])}
        if not isinstance(payload, dict):
            payload = {"output": payload}
        payload["_http_status"] = int(resp.status_code)
        tr_cont = resp.headers.get("tr_cont")
        if tr_cont is not None:
            payload["tr_cont"] = str(tr_cont)
        return payload

    def _fetch_open_orders_for_target(self, market: str) -> Dict[str, Any]:
        market_upper = str(market or "").strip().upper()
        if market_upper not in {"KR", "US"}:
            return {"market": market_upper, "ok": False, "open_orders": [], "errors": ["unsupported_market"]}
        out: Dict[str, Any] = {
            "market": market_upper,
            "ok": False,
            "open_orders": [],
            "errors": [],
            "attempted_exchanges": [],
        }
        param_candidates = [
            {"CTX_AREA_FK100": "", "CTX_AREA_NK100": "", "INQR_DVSN_1": "0", "INQR_DVSN_2": "0"},
            {"CTX_AREA_FK100": "", "CTX_AREA_NK100": "", "INQR_DVSN_1": "00", "INQR_DVSN_2": "00"},
        ]
        for exchange_code in self._exchange_candidates(market_upper):
            broker = self._get_broker(market_upper, exchange_code)
            out["attempted_exchanges"].append(exchange_code)
            if broker is None:
                out["errors"].append(f"{exchange_code}:broker_unavailable")
                continue
            exchange_ok = False
            if market_upper == "US":
                direct = self._fetch_open_orders_us_direct_exchange(exchange_code)
                if bool(direct.get("ok", False)):
                    out["open_orders"].extend(list(direct.get("open_orders", []) or []))
                    exchange_ok = True
                out["errors"].extend(list(direct.get("errors", []) or []))
                # US direct REST is preferred; keep legacy call only as fallback.
                allow_legacy = self._env_bool("AGENT_LAB_US_OPEN_ORDER_LEGACY_FALLBACK", False)
                if exchange_ok or not allow_legacy:
                    if exchange_ok:
                        out["ok"] = True
                    continue
            for idx, param in enumerate(param_candidates, start=1):
                try:
                    payload = self._call_with_rate_limit_retry(
                        lambda p=dict(param), b=broker: b.fetch_open_order(p),
                        op_name=f"{market_upper}.{exchange_code}.fetch_open_order.{idx}",
                    )
                    if not isinstance(payload, dict):
                        out["errors"].append(f"{exchange_code}:invalid_payload_type")
                        continue
                    api_err = self._api_error_text(payload)
                    if api_err:
                        out["errors"].append(f"{exchange_code}:{api_err}")
                        continue
                    rows = self._parse_open_order_rows(market=market_upper, exchange=exchange_code, payload=payload)
                    out["open_orders"].extend(rows)
                    exchange_ok = True
                    break
                except Exception as exc:
                    out["errors"].append(f"{exchange_code}:{repr(exc)}")
            if exchange_ok:
                out["ok"] = True
        if market_upper == "KR" and "KR" in out.get("attempted_exchanges", []) and not out.get("ok", False):
            out["errors"].append("KR:open_order_lookup_failed")
        if market_upper == "US":
            # SDK does not expose dedicated overseas open-order inquiry in all environments.
            # Keep lookup best-effort and avoid hard-failing sync when empty.
            out["best_effort"] = True
        return out

    def fetch_open_orders_snapshot(self, market: str = "ALL") -> Dict[str, Any]:
        scope = str(market or "ALL").strip().upper()
        targets = ["KR", "US"] if scope in {"ALL", "*"} else [scope]
        out: Dict[str, Any] = {
            "ok": True,
            "market_scope": "ALL" if len(targets) > 1 else targets[0],
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "markets": {},
            "open_orders": [],
            "errors": [],
        }
        for mk in targets:
            market_out = self._fetch_open_orders_for_target(mk)
            out["markets"][mk] = market_out
            out["open_orders"].extend(list(market_out.get("open_orders", []) or []))
            out["errors"].extend(list(market_out.get("errors", []) or []))
            if mk == "KR" and not bool(market_out.get("ok", False)):
                out["ok"] = False
        return out

    @staticmethod
    def _normalize_oversea_payload(payload: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            return [], []
        output1 = payload.get("output1", [])
        output2 = payload.get("output2", [])

        # Some APIs return a single "output" body instead of output1/output2.
        if not output1 and not output2 and "output" in payload:
            output = payload.get("output")
            if isinstance(output, list):
                output1 = output
            elif isinstance(output, dict):
                output2 = [output]

        # Some variants expose summary on output3.
        if not output2 and "output3" in payload:
            output3 = payload.get("output3")
            if isinstance(output3, list):
                output2 = output3
            elif isinstance(output3, dict):
                output2 = [output3]

        # Normalize dict -> list.
        if isinstance(output1, dict):
            output1 = [output1]
        if isinstance(output2, dict):
            output2 = [output2]

        out1: List[Dict[str, Any]] = []
        out2: List[Dict[str, Any]] = []
        for row in output1 if isinstance(output1, list) else []:
            if isinstance(row, dict):
                out1.append(row)
        for row in output2 if isinstance(output2, list) else []:
            if isinstance(row, dict):
                out2.append(row)
        return out1, out2

    @staticmethod
    def _api_error_text(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        rt_cd = str(payload.get("rt_cd", "")).strip()
        msg_cd = str(payload.get("msg_cd", "")).strip()
        msg1 = str(payload.get("msg1", "")).strip()
        if rt_cd and rt_cd != "0":
            return f"rt_cd={rt_cd}, msg_cd={msg_cd}, msg1={msg1}"
        return ""

    @staticmethod
    def _is_rate_limit_text(text: str) -> bool:
        raw = str(text or "")
        norm = raw.lower()
        return (
            ("egw00201" in norm)
            or ("초당 거래건수를 초과" in raw)
            or ("too many requests" in norm)
            or ("rate limit" in norm)
        )

    @staticmethod
    def _is_token_expired_text(text: str) -> bool:
        raw = str(text or "")
        norm = raw.lower()
        return (
            ("egw00123" in norm)
            or ("기간이 만료된 token" in raw)
            or ("token 만료" in raw)
            or ("expired token" in norm)
            or ("invalid token" in norm)
            or ("token expired" in norm)
        )

    @staticmethod
    def _is_transient_error_text(text: str) -> bool:
        raw = str(text or "")
        norm = raw.lower()
        keys = (
            "keyerror('tr_cont')",
            'keyerror("tr_cont")',
            "remote end closed connection without response",
            "remotedisconnected",
            "connection aborted",
            "max retries exceeded",
            "failed to resolve",
            "name or service not known",
            "temporary failure in name resolution",
            "read timed out",
            "connect timeout",
            "connection reset by peer",
            "sslerror",
            "protocolerror",
        )
        return any(k in norm for k in keys)

    @classmethod
    def _is_rate_limited_payload(cls, payload: Any) -> bool:
        return cls._is_rate_limit_text(cls._api_error_text(payload))

    def _refresh_broker_tokens(self, *, op_name: str, error_text: str) -> bool:
        seen_ids: Set[int] = set()
        brokers: List[Any] = []
        for broker in self._brokers.values():
            ident = id(broker)
            if ident in seen_ids:
                continue
            seen_ids.add(ident)
            brokers.append(broker)
        if not brokers:
            return False

        old_tokens: Dict[int, str] = {}
        for broker in brokers:
            old_tokens[id(broker)] = str(getattr(broker, "access_token", "") or "")

        issued = False
        errors: List[str] = []
        for broker in brokers:
            issue_fn = getattr(broker, "issue_access_token", None)
            if not callable(issue_fn):
                continue
            try:
                issue_fn()
                issued = True
                break
            except Exception as exc:
                errors.append(f"issue:{repr(exc)}")

        for broker in brokers:
            load_fn = getattr(broker, "load_access_token", None)
            if not callable(load_fn):
                continue
            try:
                load_fn()
            except Exception as exc:
                errors.append(f"load:{repr(exc)}")

        changed = False
        for broker in brokers:
            before = old_tokens.get(id(broker), "")
            after = str(getattr(broker, "access_token", "") or "")
            if after and after != before:
                changed = True
                break

        ok = bool(issued or changed)
        try:
            self.storage.log_event(
                "broker_token_refresh",
                {
                    "op": str(op_name),
                    "ok": bool(ok),
                    "issued": bool(issued),
                    "changed": bool(changed),
                    "broker_count": int(len(brokers)),
                    "error": str(error_text)[:240],
                    "errors": [str(err)[:180] for err in errors[:5]],
                },
                datetime.now().isoformat(timespec="seconds"),
            )
        except Exception:
            pass
        return ok

    @staticmethod
    def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(float(os.getenv(name, str(default)) or default))
        except Exception:
            value = int(default)
        value = max(minimum, value)
        value = min(maximum, value)
        return value

    @staticmethod
    def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(os.getenv(name, str(default)) or default)
        except Exception:
            value = float(default)
        value = max(minimum, value)
        value = min(maximum, value)
        return value

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
        if not raw:
            return bool(default)
        return raw in {"1", "true", "yes", "y", "on"}

    def _compute_us_aggressive_limit(self, *, side: str, ref_price: float) -> float | None:
        px = float(ref_price or 0.0)
        if px <= 0:
            return None
        buy_bps = self._env_float("AGENT_LAB_US_AGGRESSIVE_BUY_BPS", 120.0, 1.0, 5000.0)
        sell_bps = self._env_float("AGENT_LAB_US_AGGRESSIVE_SELL_BPS", 120.0, 1.0, 5000.0)
        side_upper = str(side or "").strip().upper()
        if side_upper == "BUY":
            out = px * (1.0 + (buy_bps / 10000.0))
            return round(max(0.0001, out), 4)
        if side_upper == "SELL":
            out = px * (1.0 - (sell_bps / 10000.0))
            return round(max(0.0001, out), 4)
        return None

    def _call_with_rate_limit_retry(self, fn: Any, *, op_name: str) -> Any:
        rate_retries = self._env_int("AGENT_LAB_BROKER_RATE_LIMIT_RETRIES", 2, 0, 10)
        rate_base_delay = self._env_float("AGENT_LAB_BROKER_RATE_LIMIT_BACKOFF_SEC", 1.2, 0.0, 60.0)
        rate_mult = self._env_float("AGENT_LAB_BROKER_RATE_LIMIT_BACKOFF_MULT", 1.7, 1.0, 5.0)
        rate_max_delay = self._env_float("AGENT_LAB_BROKER_RATE_LIMIT_BACKOFF_MAX_SEC", 8.0, 0.1, 120.0)
        transient_retries = self._env_int("AGENT_LAB_BROKER_TRANSIENT_RETRIES", 2, 0, 10)
        transient_base_delay = self._env_float("AGENT_LAB_BROKER_TRANSIENT_BACKOFF_SEC", 0.8, 0.0, 60.0)
        transient_mult = self._env_float("AGENT_LAB_BROKER_TRANSIENT_BACKOFF_MULT", 1.8, 1.0, 5.0)
        transient_max_delay = self._env_float("AGENT_LAB_BROKER_TRANSIENT_BACKOFF_MAX_SEC", 6.0, 0.1, 120.0)
        token_refresh_retries = self._env_int("AGENT_LAB_BROKER_TOKEN_REFRESH_RETRIES", 1, 0, 3)

        rate_attempt = 0
        transient_attempt = 0
        token_refresh_attempt = 0
        last_exc: Exception | None = None
        while True:
            try:
                payload = fn()
                api_err = self._api_error_text(payload)
                if self._is_token_expired_text(api_err):
                    if token_refresh_attempt < token_refresh_retries:
                        token_refresh_attempt += 1
                        if self._refresh_broker_tokens(op_name=op_name, error_text=api_err):
                            continue
                    raise RuntimeError(api_err)
                if self._is_rate_limited_payload(payload):
                    raise RuntimeError(api_err)
                return payload
            except Exception as exc:
                last_exc = exc
                err_text = repr(exc)
                is_token_expired = self._is_token_expired_text(err_text)
                is_rate = self._is_rate_limit_text(err_text)
                is_transient = self._is_transient_error_text(err_text)
                if is_token_expired and token_refresh_attempt < token_refresh_retries:
                    token_refresh_attempt += 1
                    if self._refresh_broker_tokens(op_name=op_name, error_text=err_text):
                        continue
                if is_rate and rate_attempt < rate_retries:
                    rate_attempt += 1
                    delay = min(rate_max_delay, rate_base_delay * (rate_mult ** max(0, rate_attempt - 1)))
                    try:
                        self.storage.log_event(
                            "broker_rate_limit_retry",
                            {
                                "op": str(op_name),
                                "attempt": int(rate_attempt),
                                "max_attempts": int(rate_retries + 1),
                                "delay_sec": round(float(delay), 3),
                                "error": err_text[:240],
                            },
                            datetime.now().isoformat(timespec="seconds"),
                        )
                    except Exception:
                        pass
                    time.sleep(delay)
                    continue
                if is_transient and transient_attempt < transient_retries:
                    transient_attempt += 1
                    delay = min(
                        transient_max_delay,
                        transient_base_delay * (transient_mult ** max(0, transient_attempt - 1)),
                    )
                    try:
                        self.storage.log_event(
                            "broker_transient_retry",
                            {
                                "op": str(op_name),
                                "attempt": int(transient_attempt),
                                "max_attempts": int(transient_retries + 1),
                                "delay_sec": round(float(delay), 3),
                                "error": err_text[:240],
                            },
                            datetime.now().isoformat(timespec="seconds"),
                        )
                    except Exception:
                        pass
                    time.sleep(delay)
                    continue
                try:
                    self.storage.log_event(
                        "broker_retry_exhausted",
                        {
                            "op": str(op_name),
                            "rate_attempts": int(rate_attempt),
                            "transient_attempts": int(transient_attempt),
                            "error": err_text[:240],
                        },
                        datetime.now().isoformat(timespec="seconds"),
                    )
                except Exception:
                    pass
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{op_name}:unknown_error")

    def _fetch_us_balance_with_fallback(self, broker: Any) -> Dict[str, Any]:
        """
        Primary: fetch_balance()
        Fallback: fetch_present_balance() variants for environments where
        oversea day/night endpoint payload differs (e.g. missing 'output').
        """
        try:
            payload = self._call_with_rate_limit_retry(
                lambda: broker.fetch_balance(),
                op_name="US.fetch_balance",
            )
            if isinstance(payload, dict):
                return payload
            return {}
        except Exception as exc:
            last_error = repr(exc)

        fallback_errors: List[str] = [f"fetch_balance:{last_error}"]
        candidates: List[Dict[str, Any]] = []
        for foreign_currency in (False, True):
            try:
                payload = self._call_with_rate_limit_retry(
                    lambda fc=foreign_currency: broker.fetch_present_balance(foreign_currency=fc),
                    op_name=f"US.fetch_present_balance.{int(bool(foreign_currency))}",
                )
                if isinstance(payload, dict):
                    api_err = self._api_error_text(payload)
                    if api_err:
                        fallback_errors.append(f"fetch_present_balance({foreign_currency}):{api_err}")
                        continue
                    norm1, norm2 = self._normalize_oversea_payload(payload)
                    if norm1 or norm2:
                        candidates.append(payload)
            except Exception as exc:
                fallback_errors.append(f"fetch_present_balance({foreign_currency}):{repr(exc)}")

        if candidates:
            picked = candidates[0]
            picked["_fallback_source"] = "fetch_present_balance"
            picked["_fallback_errors"] = fallback_errors
            return picked

        raise RuntimeError("; ".join(fallback_errors))

    @staticmethod
    def _merge_us_exchange_snapshots(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_symbol: Dict[str, Dict[str, Any]] = {}
        cash_candidates: List[float] = []
        equity_candidates: List[float] = []
        raw_by_exchange: Dict[str, Any] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            exchange = str(item.get("exchange", "")).strip().upper()
            parsed = item.get("parsed", {}) if isinstance(item.get("parsed"), dict) else {}
            payload = item.get("payload", {}) if isinstance(item.get("payload"), dict) else {}
            if exchange:
                raw_by_exchange[exchange] = payload
            cash = float(parsed.get("cash_krw", 0.0) or 0.0)
            equity = float(parsed.get("equity_krw", 0.0) or 0.0)
            if cash >= 0:
                cash_candidates.append(cash)
            if equity >= 0:
                equity_candidates.append(equity)
            for pos in list(parsed.get("positions", []) or []):
                if not isinstance(pos, dict):
                    continue
                symbol = str(pos.get("symbol", "")).strip().upper()
                qty = float(pos.get("quantity", 0.0) or 0.0)
                if not symbol or qty <= 0:
                    continue
                avg = float(pos.get("avg_price", 0.0) or 0.0)
                mv = float(pos.get("market_value_krw", 0.0) or 0.0)
                fx = float(pos.get("fx_rate", 1.0) or 1.0)
                ccy = str(pos.get("currency", "USD") or "USD").strip().upper()
                st = by_symbol.setdefault(
                    symbol,
                    {
                        "market": "US",
                        "symbol": symbol,
                        "quantity": 0.0,
                        "avg_cost_krw": 0.0,
                        "market_value_krw": 0.0,
                        "fx_num": 0.0,
                        "fx_den": 0.0,
                        "currency": ccy,
                    },
                )
                st["quantity"] += qty
                st["avg_cost_krw"] += qty * avg
                st["market_value_krw"] += max(0.0, mv)
                st["fx_num"] += qty * max(0.0, fx)
                st["fx_den"] += qty
        positions: List[Dict[str, Any]] = []
        for st in by_symbol.values():
            qty = float(st.get("quantity", 0.0) or 0.0)
            if qty <= 0:
                continue
            avg_price = float(st.get("avg_cost_krw", 0.0) or 0.0) / qty if qty > 0 else 0.0
            fx_rate = float(st.get("fx_num", 0.0) or 0.0) / float(st.get("fx_den", 0.0) or 1.0)
            if fx_rate <= 0:
                fx_rate = 1.0
            positions.append(
                {
                    "market": "US",
                    "symbol": str(st.get("symbol", "")).strip().upper(),
                    "quantity": qty,
                    "avg_price": avg_price,
                    "market_value_krw": float(st.get("market_value_krw", 0.0) or 0.0),
                    "currency": str(st.get("currency", "USD")),
                    "fx_rate": fx_rate,
                    "payload": {"source": "merged_us_exchange_snapshots"},
                }
            )
        cash_krw = max(cash_candidates) if cash_candidates else 0.0
        equity_krw = max(equity_candidates) if equity_candidates else cash_krw
        return {
            "market": "US",
            "cash_krw": float(cash_krw),
            "equity_krw": float(max(equity_krw, cash_krw)),
            "positions": positions,
            "raw": {"by_exchange": raw_by_exchange},
        }

    def _fetch_us_snapshot_all_exchanges(self) -> Dict[str, Any]:
        errors: List[str] = []
        snapshots: List[Dict[str, Any]] = []
        spacing = self._env_float("AGENT_LAB_US_BALANCE_EXCHANGE_SPACING_SEC", 0.25, 0.0, 10.0)
        for exchange_code in self._exchange_candidates("US"):
            broker = self._get_broker("US", exchange_code)
            if broker is None:
                errors.append(f"{exchange_code}:broker_unavailable")
                continue
            try:
                payload = self._fetch_us_balance_with_fallback(broker)
                parsed = self._parse_balance_oversea(payload if isinstance(payload, dict) else {})
                snapshots.append({"exchange": exchange_code, "payload": payload, "parsed": parsed})
            except Exception as exc:
                errors.append(f"{exchange_code}:{repr(exc)}")
            if spacing > 0:
                time.sleep(spacing)
        if not snapshots:
            raise RuntimeError("; ".join(errors) if errors else "US:all_exchange_snapshot_failed")
        merged = self._merge_us_exchange_snapshots(snapshots)
        merged["raw_errors"] = errors
        merged["exchange_count"] = len(snapshots)
        return merged

    def fetch_account_snapshot(self, market: str = "ALL") -> Dict[str, Any]:
        market_upper = str(market or "ALL").strip().upper()
        targets = ["KR", "US"] if market_upper in {"ALL", "*"} else [market_upper]
        out: Dict[str, Any] = {
            "ok": False,
            "market_scope": "ALL" if len(targets) > 1 else targets[0],
            "cash_krw": 0.0,
            "equity_krw": 0.0,
            "positions": [],
            "markets": {},
            "errors": [],
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "source": self.execution_mode,
        }

        for mk in targets:
            broker = None
            if mk != "US":
                broker = self._get_broker(mk)
                if broker is None:
                    out["errors"].append(f"{mk}:broker_unavailable")
                    continue
            try:
                if mk == "US":
                    parsed = self._fetch_us_snapshot_all_exchanges()
                else:
                    payload = self._call_with_rate_limit_retry(
                        lambda: broker.fetch_balance(),
                        op_name=f"{mk}.fetch_balance",
                    )
                    parsed = self._parse_balance_domestic(payload if isinstance(payload, dict) else {})
                out["markets"][mk] = parsed
                out["cash_krw"] += float(parsed.get("cash_krw", 0.0) or 0.0)
                out["equity_krw"] += float(parsed.get("equity_krw", 0.0) or 0.0)
                out["positions"].extend(list(parsed.get("positions", []) or []))
            except Exception as exc:
                out["errors"].append(f"{mk}:{repr(exc)}")
            spacing = self._env_float("AGENT_LAB_BROKER_BALANCE_CALL_SPACING_SEC", 0.25, 0.0, 10.0)
            if spacing > 0:
                time.sleep(spacing)

        out["ok"] = len(out["errors"]) == 0 and len(out["markets"]) == len(targets)
        return out

    def execute_orders(
        self,
        *,
        proposal_id: int,
        agent_id: str,
        market: str,
        orders: List[Dict[str, Any]],
        fx_rate: float,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        now_ts = datetime.now().isoformat(timespec="seconds")
        market_upper = str(market or "").strip().upper()
        enable_daily_blacklist = str(os.getenv("AGENT_LAB_DAILY_UNTRADABLE_BLACKLIST", "1")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        enable_sell_precheck_sync = str(os.getenv("AGENT_LAB_SELL_PRECHECK_SYNC", "1")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        sell_repeat_guard_sec = self._env_int("AGENT_LAB_SELL_REPEAT_GUARD_SEC", 900, 0, 86400)
        session_date = self._session_date_for_market(market_upper)
        daily_untradable_symbols = (
            self._load_daily_untradable_symbols(market_upper, session_date) if enable_daily_blacklist else set()
        )

        sell_precheck_sync: Dict[str, Any] = {"ok": False, "reason": "disabled"}
        sell_available: Dict[str, float] = {}
        if self.execution_mode in {"mojito_mock", "mock_api"} and enable_sell_precheck_sync:
            try:
                sell_precheck_sync = self.fetch_account_snapshot(market=market_upper)
            except Exception as exc:
                sell_precheck_sync = {"ok": False, "reason": "exception", "errors": [repr(exc)]}
            if bool(sell_precheck_sync.get("ok", False)):
                for pos in list(sell_precheck_sync.get("positions", []) or []):
                    if not isinstance(pos, dict):
                        continue
                    if str(pos.get("market", "")).strip().upper() != market_upper:
                        continue
                    sym = str(pos.get("symbol", "")).strip().upper()
                    qty = float(pos.get("quantity", 0.0) or 0.0)
                    if not sym or qty <= 0:
                        continue
                    sell_available[sym] = sell_available.get(sym, 0.0) + qty

        for order in orders:
            symbol = str(order["symbol"]).strip().upper()
            side = str(order.get("side", "BUY")).upper()
            qty = float(order.get("quantity", 0.0) or 0.0)
            order_type = str(order.get("order_type", "MARKET")).upper()
            limit_price = order.get("limit_price")
            ref_price = float(order.get("reference_price", 0.0) or 0.0)
            if qty <= 0 or ref_price <= 0:
                continue

            broker_order_id = ""
            broker_resp: Dict[str, Any] = {}
            status = "FILLED"
            submit_qty = float(qty)
            if self.execution_mode in {"mojito_mock", "mock_api"}:
                precheck_reject_reason = ""
                if enable_daily_blacklist and symbol in daily_untradable_symbols:
                    precheck_reject_reason = "daily_untradable_blacklist"
                elif side == "SELL":
                    if self._has_recent_sell_submission(market_upper, symbol, sell_repeat_guard_sec):
                        precheck_reject_reason = f"recent_sell_submission_guard({sell_repeat_guard_sec}s)"
                    elif enable_sell_precheck_sync and not bool(sell_precheck_sync.get("ok", False)):
                        precheck_reject_reason = "sell_precheck_sync_failed"
                    elif enable_sell_precheck_sync:
                        available_qty = float(sell_available.get(symbol, 0.0) or 0.0)
                        if available_qty <= 0:
                            precheck_reject_reason = "sell_precheck_no_position"
                        elif submit_qty > available_qty:
                            submit_qty = float(available_qty)

                if precheck_reject_reason:
                    status = "REJECTED"
                    broker_resp = {
                        "ok": False,
                        "reason": precheck_reject_reason,
                        "market": market_upper,
                        "symbol": symbol,
                        "side": side,
                        "requested_quantity": float(qty),
                        "submit_quantity": float(submit_qty),
                        "sell_precheck_sync": sell_precheck_sync if side == "SELL" and enable_sell_precheck_sync else {},
                        "fx_rate": float(fx_rate),
                    }
                else:
                    send_order = dict(order)
                    send_order["quantity"] = float(submit_qty)
                    send = self._send_order(market=market_upper, order=send_order)
                    broker_resp = dict(send)
                    broker_resp["fx_rate"] = float(fx_rate)
                    broker_resp["requested_quantity"] = float(qty)
                    broker_resp["submit_quantity"] = float(submit_qty)
                    if send.get("ok"):
                        payload = send.get("response")
                        api_err = self._api_error_text(payload if isinstance(payload, dict) else {})
                        if api_err:
                            status = "REJECTED"
                            broker_resp = dict(send)
                            broker_resp["ok"] = False
                            broker_resp["reason"] = api_err
                        elif isinstance(payload, dict):
                            broker_order_id = str(payload.get("ODNO") or payload.get("output", {}).get("ODNO") or "")
                            status = "SUBMITTED"
                    else:
                        status = "REJECTED"
                    eff_type = str(send.get("effective_order_type", order_type) or order_type).strip().upper()
                    if eff_type in {"LIMIT", "MARKET"}:
                        order_type = eff_type
                    eff_limit = send.get("effective_limit_price")
                    if eff_limit is not None:
                        try:
                            limit_price = float(eff_limit)
                        except Exception:
                            pass

                if side == "SELL" and status in {"SUBMITTED", "FILLED"}:
                    current = float(sell_available.get(symbol, 0.0) or 0.0)
                    sell_available[symbol] = max(0.0, current - float(submit_qty))

                if enable_daily_blacklist and status == "REJECTED":
                    err_text = self._collect_send_error_text(broker_resp)
                    if self._is_non_tradable_error_text(err_text):
                        added = self._append_daily_untradable_symbol(market_upper, session_date, symbol)
                        daily_untradable_symbols.add(symbol)
                        if added:
                            self.storage.log_event(
                                "daily_symbol_blacklisted",
                                {
                                    "market": market_upper,
                                    "session_date": session_date,
                                    "symbol": symbol,
                                    "reason": err_text[:240],
                                },
                                now_ts,
                            )

            paper_order_id = self.storage.insert_paper_order(
                proposal_id=proposal_id,
                agent_id=agent_id,
                market=market_upper,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=float(submit_qty),
                limit_price=None if limit_price is None else float(limit_price),
                reference_price=ref_price,
                status=status,
                broker_order_id=broker_order_id,
                broker_response_json=broker_resp,
                submitted_at=now_ts,
            )

            if status == "FILLED":
                fill_price = ref_price if order_type != "LIMIT" else float(limit_price or ref_price)
                fill_value_krw = float(fill_price) * float(submit_qty) * float(fx_rate)
                self.storage.insert_paper_fill(
                    paper_order_id=paper_order_id,
                    fill_price=float(fill_price),
                    fill_quantity=float(submit_qty),
                    fill_value_krw=float(fill_value_krw),
                    fx_rate=float(fx_rate),
                    filled_at=now_ts,
                )
            results.append(
                {
                    "paper_order_id": paper_order_id,
                    "symbol": symbol,
                    "side": side,
                    "status": status,
                    "reference_price": ref_price,
                    "quantity": float(submit_qty),
                    "fx_rate": fx_rate,
                    "reject_reason": self._extract_reject_reason(broker_resp) if status == "REJECTED" else "",
                }
            )
        return results
