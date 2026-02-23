from __future__ import annotations

import importlib
import os
from datetime import datetime
from typing import Any, Dict, List

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
        symbol = str(order["symbol"])
        qty = int(float(order["quantity"]))
        order_type = str(order.get("order_type", "MARKET")).upper()
        side = str(order.get("side", "BUY")).upper()
        attempts: List[Dict[str, Any]] = []
        for exchange_code in self._exchange_candidates(market):
            broker = self._get_broker(market, exchange_code)
            if broker is None:
                attempts.append({"exchange": exchange_code, "ok": False, "reason": "broker_unavailable"})
                continue
            try:
                if order_type == "LIMIT":
                    px = int(float(order.get("limit_price") or order.get("reference_price") or 0))
                    if side == "BUY":
                        resp = broker.create_limit_buy_order(symbol, px, qty)
                    else:
                        resp = broker.create_limit_sell_order(symbol, px, qty)
                else:
                    if side == "BUY":
                        resp = broker.create_market_buy_order(symbol, qty)
                    else:
                        resp = broker.create_market_sell_order(symbol, qty)
                return {"ok": True, "response": resp, "exchange": exchange_code, "attempts": attempts}
            except Exception as exc:
                attempts.append({"exchange": exchange_code, "ok": False, "reason": repr(exc)})
                continue
        return {"ok": False, "reason": "all_exchange_attempts_failed", "attempts": attempts}

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

        cash_krw = cls._pick_value(summary, ["dnca_tot_amt", "ord_psbl_cash", "dnca_tot_amt2", "tot_dnca"])
        equity_krw = cls._pick_value(summary, ["tot_evlu_amt", "scts_evlu_amt", "tot_asst_amt", "tot_eval_amt"], default=cash_krw)
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
        return {
            "market": "KR",
            "cash_krw": float(cash_krw),
            "equity_krw": float(max(equity_krw, cash_krw)),
            "positions": positions,
            "raw": payload,
        }

    @classmethod
    def _parse_balance_oversea(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        output1 = payload.get("output1", [])
        output2 = payload.get("output2", [])
        if not isinstance(output1, list):
            output1 = []
        if not isinstance(output2, list):
            output2 = []
        summary = output2[0] if output2 else {}
        if not isinstance(summary, dict):
            summary = {}

        # Overseas API may return local-currency values. Prefer KRW fields if present.
        cash_krw = cls._pick_value(summary, ["frcr_evlu_pfls_amt", "tot_evlu_pfls_amt", "ovrs_ord_psbl_amt", "cash_amt"])
        equity_krw = cls._pick_value(summary, ["tot_evlu_amt", "frcr_buy_amt_smtl", "ovrs_tot_pfls", "tot_asst_amt"], default=cash_krw)
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
            qty = cls._pick_value(row, ["ovrs_cblc_qty", "hldg_qty", "ord_psbl_qty", "qty"])
            if not symbol or qty <= 0:
                continue
            avg_price = cls._pick_value(row, ["pchs_avg_pric", "avg_pric", "avg_price"])
            fx_rate = cls._pick_value(row, ["bass_exrt", "frcr_exrt", "fx_rate"], default=1.0)
            if fx_rate <= 0:
                fx_rate = 1.0
            market_value_krw = cls._pick_value(
                row,
                ["frcr_evlu_amt2", "evlu_pfls_amt", "evlu_amt", "market_value_krw"],
            )
            if market_value_krw <= 0:
                local_px = cls._pick_value(row, ["now_pric2", "ovrs_now_pric1", "last", "close"])
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
        return {
            "market": "US",
            "cash_krw": float(cash_krw),
            "equity_krw": float(max(equity_krw, cash_krw)),
            "positions": positions,
            "raw": payload,
        }

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
            broker = self._get_broker(mk)
            if broker is None:
                out["errors"].append(f"{mk}:broker_unavailable")
                continue
            try:
                payload = broker.fetch_balance()
                if mk == "KR":
                    parsed = self._parse_balance_domestic(payload if isinstance(payload, dict) else {})
                else:
                    parsed = self._parse_balance_oversea(payload if isinstance(payload, dict) else {})
                out["markets"][mk] = parsed
                out["cash_krw"] += float(parsed.get("cash_krw", 0.0) or 0.0)
                out["equity_krw"] += float(parsed.get("equity_krw", 0.0) or 0.0)
                out["positions"].extend(list(parsed.get("positions", []) or []))
            except Exception as exc:
                out["errors"].append(f"{mk}:{repr(exc)}")

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
        for order in orders:
            symbol = str(order["symbol"])
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
            if self.execution_mode in {"mojito_mock", "mock_api"}:
                send = self._send_order(market=market, order=order)
                broker_resp = send
                if send.get("ok"):
                    payload = send.get("response")
                    if isinstance(payload, dict):
                        broker_order_id = str(payload.get("ODNO") or payload.get("output", {}).get("ODNO") or "")
                else:
                    status = "REJECTED"

            paper_order_id = self.storage.insert_paper_order(
                proposal_id=proposal_id,
                agent_id=agent_id,
                market=market,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=qty,
                limit_price=None if limit_price is None else float(limit_price),
                reference_price=ref_price,
                status=status,
                broker_order_id=broker_order_id,
                broker_response_json=broker_resp,
                submitted_at=now_ts,
            )

            if status == "FILLED":
                fill_price = ref_price if order_type != "LIMIT" else float(limit_price or ref_price)
                fill_value_krw = float(fill_price) * float(qty) * float(fx_rate)
                self.storage.insert_paper_fill(
                    paper_order_id=paper_order_id,
                    fill_price=float(fill_price),
                    fill_quantity=float(qty),
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
                    "quantity": qty,
                    "fx_rate": fx_rate,
                }
            )
        return results
