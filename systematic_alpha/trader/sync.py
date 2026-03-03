from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None  # type: ignore

from systematic_alpha.credentials import load_credentials
from systematic_alpha.helpers import to_float
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.trader.config import TraderConfig, state_dir_from_cfg
from systematic_alpha.trader.storage import TraderStorage


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _pick(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return default


def _latest_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("output1", "output"):
        val = payload.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    return []


def _api_error_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "non_dict_payload"
    rt_cd = str(payload.get("rt_cd", "") or "").strip()
    msg_cd = str(payload.get("msg_cd", "") or "").strip()
    msg1 = str(payload.get("msg1", "") or "").strip()
    if rt_cd and rt_cd != "0":
        chunks = [f"rt_cd={rt_cd}"]
        if msg_cd:
            chunks.append(f"msg_cd={msg_cd}")
        if msg1:
            chunks.append(f"msg1={msg1}")
        return ", ".join(chunks)
    if msg_cd and msg_cd != "0":
        chunks = [f"msg_cd={msg_cd}"]
        if msg1:
            chunks.append(f"msg1={msg1}")
        return ", ".join(chunks)
    if "output" not in payload and "output1" not in payload:
        if msg1:
            return f"missing_output:{msg1}"
        return "missing_output"
    return ""


def _is_rate_limit_error(text: str) -> bool:
    t = str(text or "").lower()
    return "egw00201" in t or "초당 거래건수를 초과" in t or "rate limit" in t


def _is_token_error(text: str) -> bool:
    t = str(text or "").lower()
    return "egw00123" in t or "기간이 만료된 token" in t


@dataclass
class SyncResult:
    market_scope: str
    strict: bool
    ok: bool
    blocked: bool
    reason: str
    errors: List[str]
    cash_krw: float
    equity_krw: float
    positions: List[Dict[str, Any]]
    snapshot_id: int

    def as_payload(self) -> Dict[str, Any]:
        return {
            "market_scope": self.market_scope,
            "strict": self.strict,
            "ok": self.ok,
            "blocked": self.blocked,
            "reason": self.reason,
            "errors": self.errors,
            "cash_krw": self.cash_krw,
            "equity_krw": self.equity_krw,
            "position_count": len([p for p in self.positions if float(p.get("quantity", 0) or 0) > 0]),
            "snapshot_id": self.snapshot_id,
        }


class AccountSyncService:
    def __init__(self, config: TraderConfig, storage: TraderStorage):
        self.config = config
        self.storage = storage
        self._brokers: Dict[str, Any] = {}
        self._last_call_ts = 0.0
        self._lock_fd = None
        self._creds_loaded = False
        self._creds: Dict[str, str] = {}

        runtime_dir = state_dir_from_cfg(self.config) / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = runtime_dir / "broker_global.lock"

        # Keep mojito dayornight spacing in sync with trader config.
        os.environ.setdefault(
            "TRADER_US_DAYORNIGHT_CALL_SPACING_SEC",
            str(self.config.us_dayornight_call_spacing_sec),
        )

    def _load_creds(self) -> None:
        if self._creds_loaded:
            return
        self._creds_loaded = True
        key, secret, acc_no, _ = load_credentials(None)
        self._creds = {"key": key, "secret": secret, "acc_no": acc_no}

    def _broker(self, market: str, exchange_code: str = ""):
        self._load_creds()
        mk = str(market or "").upper()
        ex = str(exchange_code or "").upper()
        cache_key = f"{mk}:{ex}" if mk == "US" else mk
        if cache_key in self._brokers:
            return self._brokers[cache_key]

        mojito = import_mojito_module()
        if mk == "KR":
            broker = mojito.KoreaInvestment(
                api_key=self._creds["key"],
                api_secret=self._creds["secret"],
                acc_no=self._creds["acc_no"],
                mock=self.config.use_mock,
            )
        else:
            label = {"NASD": "나스닥", "NYSE": "뉴욕", "AMEX": "아멕스"}.get(ex, "뉴욕")
            broker = mojito.KoreaInvestment(
                api_key=self._creds["key"],
                api_secret=self._creds["secret"],
                acc_no=self._creds["acc_no"],
                exchange=label,
                mock=self.config.use_mock,
            )
        self._brokers[cache_key] = broker
        return broker

    def _acquire_global_lock(self) -> None:
        if not self.config.broker_global_serialize:
            return
        if fcntl is None:
            return
        self._lock_fd = open(self._lock_path, "a+", encoding="utf-8")
        fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)

    def _release_global_lock(self) -> None:
        if self._lock_fd is None or fcntl is None:
            return
        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._lock_fd.close()
        except Exception:
            pass
        self._lock_fd = None

    def _global_spacing(self) -> None:
        if self.config.broker_global_min_interval_sec <= 0:
            return
        now = time.time()
        elapsed = now - self._last_call_ts
        gap = self.config.broker_global_min_interval_sec
        if elapsed < gap:
            time.sleep(max(0.0, gap - elapsed))
        self._last_call_ts = time.time()

    def _call(self, fn: Any, op_name: str) -> Dict[str, Any]:
        retries = max(0, int(self.config.rate_limit_retries))
        delay = float(self.config.rate_limit_backoff_sec)
        max_delay = float(self.config.rate_limit_backoff_max_sec)
        attempt = 0
        while True:
            attempt += 1
            self._acquire_global_lock()
            try:
                self._global_spacing()
                payload = fn()
            except Exception as exc:
                payload = {"rt_cd": "1", "msg1": repr(exc)}
            finally:
                self._release_global_lock()

            err = _api_error_text(payload)
            if not err:
                return payload

            rate_or_token = _is_rate_limit_error(err) or _is_token_error(err)
            if not rate_or_token or attempt > retries + 1:
                raise RuntimeError(f"{op_name}:{err}")
            sleep_for = min(max_delay, delay * (2 ** (attempt - 1)))
            time.sleep(max(0.0, sleep_for))

    @staticmethod
    def _parse_kr(payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = _latest_list(payload)
        positions: List[Dict[str, Any]] = []
        for row in rows:
            symbol = str(_pick(row, ["pdno", "mksc_shrn_iscd", "symbol"], "") or "").strip().upper()
            if not symbol:
                continue
            qty = float(to_float(_pick(row, ["hldg_qty", "hold_qty", "qty"], 0.0)) or 0.0)
            if qty <= 0:
                continue
            avg_price = float(to_float(_pick(row, ["pchs_avg_pric", "avg_unpr", "avg_price"], 0.0)) or 0.0)
            market_value = float(to_float(_pick(row, ["evlu_amt", "market_value", "evlu_pfls_amt"], avg_price * qty)) or 0.0)
            positions.append(
                {
                    "market": "KR",
                    "symbol": symbol,
                    "quantity": qty,
                    "avg_price": avg_price,
                    "market_value_krw": max(0.0, market_value),
                    "currency": "KRW",
                    "fx_rate": 1.0,
                    "payload": row,
                }
            )

        output2 = payload.get("output2", {}) if isinstance(payload.get("output2"), dict) else {}
        cash = float(to_float(_pick(output2, ["dnca_tot_amt", "dnca_tot_amt2", "cash", "tot_evlu_amt"], 0.0)) or 0.0)
        equity = float(to_float(_pick(output2, ["tot_evlu_amt", "equity", "scts_evlu_amt"], 0.0)) or 0.0)
        if equity <= 0:
            equity = float(cash + sum(float(x.get("market_value_krw", 0.0) or 0.0) for x in positions))
        return {"cash_krw": cash, "equity_krw": equity, "positions": positions}

    @staticmethod
    def _parse_us(payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = payload.get("output1", []) if isinstance(payload.get("output1"), list) else []
        positions: List[Dict[str, Any]] = []
        for row in rows:
            symbol = str(_pick(row, ["ovrs_pdno", "pdno", "symb", "symbol"], "") or "").strip().upper()
            if not symbol:
                continue
            qty = float(
                to_float(_pick(row, ["ovrs_cblc_qty", "cblc_qty13", "hldg_qty", "ord_psbl_qty1", "qty"], 0.0))
                or 0.0
            )
            if qty <= 0:
                continue
            avg_price = float(to_float(_pick(row, ["avg_unpr3", "pchs_avg_pric", "avg_price"], 0.0)) or 0.0)
            local_eval = float(to_float(_pick(row, ["ovrs_now_evlu_pfls_amt", "frcr_evlu_amt2", "evlu_amt"], 0.0)) or 0.0)
            fx = float(to_float(_pick(row, ["bass_exrt", "frst_bltn_exrt", "fx_rate"], 1300.0)) or 1300.0)
            market_value_krw = local_eval * fx if local_eval > 0 else max(0.0, qty * avg_price * fx)
            positions.append(
                {
                    "market": "US",
                    "symbol": symbol,
                    "quantity": qty,
                    "avg_price": avg_price,
                    "market_value_krw": market_value_krw,
                    "currency": str(_pick(row, ["tr_crcy_cd", "crcy_cd"], "USD") or "USD").upper(),
                    "fx_rate": fx,
                    "payload": row,
                }
            )

        output2 = payload.get("output2", {}) if isinstance(payload.get("output2"), dict) else {}
        cash = float(to_float(_pick(output2, ["frcr_dncl_amt_2", "frcr_dncl_amt", "cash"], 0.0)) or 0.0)
        equity = float(to_float(_pick(output2, ["tot_evlu_pfls_amt", "equity", "tot_asst_amt"], 0.0)) or 0.0)
        if equity <= 0:
            equity = float(cash + sum(float(x.get("market_value_krw", 0.0) or 0.0) for x in positions))
        return {"cash_krw": cash, "equity_krw": equity, "positions": positions}

    @staticmethod
    def _merge_us(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        cash = 0.0
        equity = 0.0
        by_symbol: Dict[str, Dict[str, Any]] = {}
        for snap in snapshots:
            cash = max(cash, float(snap.get("cash_krw", 0.0) or 0.0))
            equity = max(equity, float(snap.get("equity_krw", 0.0) or 0.0))
            for pos in list(snap.get("positions", []) or []):
                symbol = str(pos.get("symbol", "") or "").upper()
                if not symbol:
                    continue
                existing = by_symbol.get(symbol)
                if existing is None:
                    by_symbol[symbol] = dict(pos)
                    continue
                # keep larger quantity snapshot to avoid unstable partial snapshots
                if float(pos.get("quantity", 0.0) or 0.0) > float(existing.get("quantity", 0.0) or 0.0):
                    by_symbol[symbol] = dict(pos)
        positions = list(by_symbol.values())
        if equity <= 0:
            equity = cash + sum(float(p.get("market_value_krw", 0.0) or 0.0) for p in positions)
        return {"cash_krw": cash, "equity_krw": equity, "positions": positions}

    def _fetch_kr(self) -> Dict[str, Any]:
        broker = self._broker("KR")
        payload = self._call(lambda b=broker: b.fetch_balance(), "KR.fetch_balance")
        return self._parse_kr(payload)

    def _fetch_us_exchange(self, exchange_code: str) -> Dict[str, Any]:
        broker = self._broker("US", exchange_code)
        payload = self._call(lambda b=broker: b.fetch_balance(), f"US.{exchange_code}.fetch_balance")
        return self._parse_us(payload)

    def sync_account(self, market_scope: str = "ALL", strict: bool = True, trade_date: str | None = None) -> SyncResult:
        scope = str(market_scope or "ALL").upper()
        strict_mode = bool(strict)
        date_txt = str(trade_date or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d"))

        targets = ["KR", "US"] if scope in {"ALL", "*"} else [scope]
        all_positions: List[Dict[str, Any]] = []
        errors: List[str] = []
        total_cash = 0.0
        total_equity = 0.0

        for mk in targets:
            if mk == "KR":
                try:
                    parsed = self._fetch_kr()
                    total_cash += float(parsed.get("cash_krw", 0.0) or 0.0)
                    total_equity += float(parsed.get("equity_krw", 0.0) or 0.0)
                    all_positions.extend(list(parsed.get("positions", []) or []))
                except Exception as exc:
                    errors.append(f"KR:{repr(exc)}")
                continue

            if mk == "US":
                exchange_results: List[Dict[str, Any]] = []
                ex_errors: List[str] = []
                for ex in list(self.config.us_sync_exchanges):
                    try:
                        res = self._fetch_us_exchange(ex)
                        exchange_results.append(res)
                    except Exception as exc:
                        ex_errors.append(f"{ex}:{repr(exc)}")
                    if self.config.us_exchange_spacing_sec > 0:
                        time.sleep(self.config.us_exchange_spacing_sec)

                if ex_errors and (self.config.us_require_all_exchanges or not exchange_results):
                    errors.append("US:" + "; ".join(ex_errors))
                if exchange_results:
                    merged = self._merge_us(exchange_results)
                    total_cash += float(merged.get("cash_krw", 0.0) or 0.0)
                    total_equity += float(merged.get("equity_krw", 0.0) or 0.0)
                    all_positions.extend(list(merged.get("positions", []) or []))
                continue

        ok = len(errors) == 0
        blocked = strict_mode and not ok
        reason = ""
        if not ok:
            joined = " ".join(errors)
            if _is_rate_limit_error(joined):
                reason = "broker_rate_limited"
            elif _is_token_error(joined):
                reason = "broker_token_expired"
            else:
                reason = "broker_fetch_failed"

        payload = {
            "market_scope": scope,
            "strict": strict_mode,
            "ok": ok,
            "blocked": blocked,
            "errors": list(errors),
            "reason": reason,
        }
        snapshot_id = self.storage.insert_account_snapshot(
            trade_date=date_txt,
            market_scope=scope,
            source="kis_openapi",
            strict=strict_mode,
            ok=ok,
            blocked=blocked,
            reason=reason,
            cash_krw=total_cash,
            equity_krw=total_equity,
            payload=payload,
            positions=all_positions,
            created_at=_now_iso(),
        )

        return SyncResult(
            market_scope=scope,
            strict=strict_mode,
            ok=ok,
            blocked=blocked,
            reason=reason,
            errors=errors,
            cash_krw=total_cash,
            equity_krw=total_equity,
            positions=all_positions,
            snapshot_id=snapshot_id,
        )


def summarize_positions(positions: List[Dict[str, Any]], market: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mk = str(market or "").upper()
    for row in positions:
        if str(row.get("market", "")).upper() != mk:
            continue
        sym = str(row.get("symbol", "")).upper()
        qty = float(row.get("quantity", 0.0) or 0.0)
        if qty <= 0:
            continue
        out[sym] = out.get(sym, 0.0) + qty
    return out
