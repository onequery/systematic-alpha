from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from systematic_alpha.helpers import latest_list_of_dict, normalize_yyyymmdd, pick_first, to_float
from systematic_alpha.models import Stage1Candidate
from systematic_alpha.trader.config import TraderConfig
from systematic_alpha.trader.selector_bridge import make_selector
from systematic_alpha.trader.storage import TraderStorage


KST = ZoneInfo("Asia/Seoul")


def _now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def _log_index_api_call(
    *,
    market: str,
    symbol: str,
    status: str,
    attempt: int,
    exchange: str = "",
    detail: str = "",
) -> None:
    msg = (
        f"[api-call] market={str(market).upper()} kind=fetch_index_ohlcv "
        f"code={str(symbol).upper()} attempt={attempt} status={status}"
    )
    if exchange:
        msg += f" exchange={exchange}"
    if detail:
        msg += f" detail={detail}"
    print(msg, flush=True)


def expected_candidate_count(cfg: TraderConfig, market: str) -> int:
    mk = str(market or "").upper()
    try:
        value = cfg.candidates_max_kr if mk == "KR" else cfg.candidates_max_us
        return max(1, int(value))
    except Exception:
        return 20


def final_candidate_cache_path(cfg: TraderConfig, market: str, trade_date: str) -> Path:
    mk = str(market or "").upper()
    slug = str(mk).lower()
    return cfg.out_dir / slug / trade_date / "cache" / f"{slug}_final_candidates.csv"


def final_candidate_cache_count(path: Path) -> int:
    csv_path = Path(path)
    if not csv_path.exists():
        return 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return sum(1 for row in reader if str(row.get("symbol", "")).strip())
    except Exception:
        return 0


def _write_final_candidate_cache(path: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trade_date",
        "market",
        "candidate_rank",
        "symbol",
        "name",
        "today_open",
        "prev_high",
        "prev_low",
        "breakout_price",
        "last_price",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "trade_date": row.get("trade_date", ""),
                    "market": row.get("market", ""),
                    "candidate_rank": row.get("candidate_rank", 0),
                    "symbol": row.get("symbol", ""),
                    "name": row.get("name", ""),
                    "today_open": row.get("today_open", ""),
                    "prev_high": row.get("prev_high", ""),
                    "prev_low": row.get("prev_low", ""),
                    "breakout_price": row.get("breakout_price", ""),
                    "last_price": row.get("last_price", ""),
                }
            )


def _parse_ohlcv_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for row in raw_rows:
        date = normalize_yyyymmdd(
            pick_first(
                row,
                (
                    "stck_bsop_date",
                    "bsop_date",
                    "stck_bsop_dt",
                    "bas_dt",
                    "xymd",
                    "date",
                    "trdt_ymd",
                    "biz_dt",
                ),
            )
        )
        if not date:
            continue
        open_price = to_float(pick_first(row, ("stck_oprc", "open", "oprc", "ovrs_oprc")))
        high_price = to_float(pick_first(row, ("stck_hgpr", "high", "hgpr", "ovrs_hgpr")))
        low_price = to_float(pick_first(row, ("stck_lwpr", "low", "lwpr", "ovrs_lwpr")))
        close_price = to_float(
            pick_first(row, ("stck_clpr", "close", "clpr", "clos", "last", "ovrs_nmix_prpr"))
        )
        if close_price is None:
            continue
        parsed.append(
            {
                "date": date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
        )
    parsed.sort(key=lambda x: str(x["date"]))
    return parsed


def _fetch_ohlcv_rows(selector: Any, market: str, symbol: str) -> List[Dict[str, Any]]:
    mk = str(market).upper()
    if mk == "KR":
        try:
            resp = selector.broker.fetch_ohlcv_recent30(symbol, timeframe="D", adj_price=True)
            return _parse_ohlcv_rows(latest_list_of_dict(resp if isinstance(resp, dict) else {}))
        except Exception:
            return []

    # US selector has exchange retry internals; use them when available.
    try:
        exchange_order = list(selector._exchange_attempt_order(symbol))  # type: ignore[attr-defined]
    except Exception:
        exchange_order = ["NYSE", "AMEX", "NASD"]

    for ex in exchange_order:
        try:
            broker = selector._get_broker_for_exchange(ex)  # type: ignore[attr-defined]
            if broker is None:
                continue
            resp = selector._fetch_ohlcv_oversea_compatible(broker, symbol)  # type: ignore[attr-defined]
            rows = _parse_ohlcv_rows(latest_list_of_dict(resp if isinstance(resp, dict) else {}))
            if rows:
                return rows
        except Exception:
            continue
    return []


def _payload_api_error(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
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
    return ""


def _is_rate_limited_error(err_text: str) -> bool:
    text = str(err_text or "").lower()
    return ("egw00201" in text) or ("초당 거래건수를 초과" in text) or ("rate limit" in text)


def _prioritized_us_exchange_order(symbol: str, order: List[str]) -> List[str]:
    sym = str(symbol or "").upper()
    preferred_map = {
        "SPY": "AMEX",
        "QQQ": "NASD",
        "DIA": "NYSE",
        "IWM": "AMEX",
    }
    preferred = preferred_map.get(sym)
    if not preferred:
        return list(order)
    # For index/ETF symbols used as regime gauges, querying the canonical exchange first
    # (and only) reduces noisy empty responses and extra API calls that can trigger rate limits.
    strict_primary = str(os.getenv("TRADER_US_OHLCV_STRICT_PRIMARY_EXCHANGE", "1") or "1").strip().lower()
    if strict_primary in {"1", "true", "yes", "on"}:
        return [preferred]
    out: List[str] = [preferred]
    out.extend([x for x in order if str(x).upper() != preferred])
    # keep unique and uppercase
    uniq: List[str] = []
    seen = set()
    for ex in out:
        exu = str(ex or "").upper()
        if not exu or exu in seen:
            continue
        seen.add(exu)
        uniq.append(exu)
    return uniq


def _fetch_ohlcv_rows_with_diag(selector: Any, market: str, symbol: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mk = str(market).upper()
    sym = str(symbol or "").upper()
    if mk == "KR":
        try:
            _log_index_api_call(
                market=mk,
                symbol=sym,
                status="start",
                attempt=1,
            )
            resp = selector.broker.fetch_ohlcv_recent30(sym, timeframe="D", adj_price=True)
            rows = _parse_ohlcv_rows(latest_list_of_dict(resp if isinstance(resp, dict) else {}))
            err = _payload_api_error(resp)
            if rows:
                _log_index_api_call(
                    market=mk,
                    symbol=sym,
                    status="ok",
                    attempt=1,
                    detail=f"bars={len(rows)}",
                )
                return rows, {"status": "ok", "market": mk, "symbol": sym, "bars": len(rows)}
            if err:
                _log_index_api_call(
                    market=mk,
                    symbol=sym,
                    status="response_error",
                    attempt=1,
                    detail=err,
                )
                return [], {"status": "fetch_failed", "market": mk, "symbol": sym, "error": err}
            _log_index_api_call(
                market=mk,
                symbol=sym,
                status="empty",
                attempt=1,
                detail="empty_ohlcv_rows",
            )
            return [], {
                "status": "empty",
                "market": mk,
                "symbol": sym,
                "error": "empty_ohlcv_rows",
            }
        except Exception as exc:
            _log_index_api_call(
                market=mk,
                symbol=sym,
                status="exception",
                attempt=1,
                detail=repr(exc),
            )
            return [], {
                "status": "fetch_failed",
                "market": mk,
                "symbol": sym,
                "error": repr(exc),
            }

    attempts: List[Dict[str, Any]] = []
    try:
        exchange_order = list(selector._exchange_attempt_order(sym))  # type: ignore[attr-defined]
    except Exception:
        exchange_order = ["NYSE", "AMEX", "NASD"]
    exchange_order = _prioritized_us_exchange_order(sym, exchange_order)

    retry_count = 2
    retry_base_sec = 1.2
    exchange_spacing_sec = 0.8
    try:
        retry_count = max(0, int(float(os.getenv("TRADER_US_OHLCV_RATE_RETRIES", "2") or 2)))
    except Exception:
        retry_count = 2
    try:
        retry_base_sec = max(0.1, float(os.getenv("TRADER_US_OHLCV_RATE_BACKOFF_SEC", "1.2") or 1.2))
    except Exception:
        retry_base_sec = 1.2
    try:
        exchange_spacing_sec = max(0.0, float(os.getenv("TRADER_US_OHLCV_EXCHANGE_SPACING_SEC", "0.8") or 0.8))
    except Exception:
        exchange_spacing_sec = 0.8

    for ex_idx, ex in enumerate(exchange_order):
        broker = None
        if ex_idx > 0 and exchange_spacing_sec > 0:
            time.sleep(exchange_spacing_sec)
        try:
            broker = selector._get_broker_for_exchange(ex)  # type: ignore[attr-defined]
            if broker is None:
                attempts.append({"exchange": ex, "status": "broker_none"})
                continue

            ex_done = False
            for attempt_idx in range(retry_count + 1):
                _log_index_api_call(
                    market=mk,
                    symbol=sym,
                    status="start",
                    attempt=attempt_idx + 1,
                    exchange=str(ex),
                )
                resp = selector._fetch_ohlcv_oversea_compatible(broker, sym)  # type: ignore[attr-defined]
                rows = _parse_ohlcv_rows(latest_list_of_dict(resp if isinstance(resp, dict) else {}))
                err = _payload_api_error(resp)
                if rows:
                    _log_index_api_call(
                        market=mk,
                        symbol=sym,
                        status="ok",
                        attempt=attempt_idx + 1,
                        exchange=str(ex),
                        detail=f"bars={len(rows)}",
                    )
                    attempts.append(
                        {
                            "exchange": ex,
                            "status": "ok",
                            "bars": len(rows),
                            "attempt": attempt_idx + 1,
                        }
                    )
                    return rows, {
                        "status": "ok",
                        "market": mk,
                        "symbol": sym,
                        "selected_exchange": ex,
                        "bars": len(rows),
                        "attempts": attempts,
                    }

                if err and _is_rate_limited_error(err):
                    _log_index_api_call(
                        market=mk,
                        symbol=sym,
                        status="response_error",
                        attempt=attempt_idx + 1,
                        exchange=str(ex),
                        detail=err,
                    )
                    attempts.append(
                        {
                            "exchange": ex,
                            "status": "rate_limited",
                            "error": err,
                            "attempt": attempt_idx + 1,
                        }
                    )
                    if attempt_idx < retry_count:
                        time.sleep(retry_base_sec * (2**attempt_idx))
                        continue
                    ex_done = True
                    break

                if err:
                    _log_index_api_call(
                        market=mk,
                        symbol=sym,
                        status="response_error",
                        attempt=attempt_idx + 1,
                        exchange=str(ex),
                        detail=err,
                    )
                    attempts.append(
                        {
                            "exchange": ex,
                            "status": "api_error",
                            "error": err,
                            "attempt": attempt_idx + 1,
                        }
                    )
                else:
                    _log_index_api_call(
                        market=mk,
                        symbol=sym,
                        status="empty",
                        attempt=attempt_idx + 1,
                        exchange=str(ex),
                        detail="empty_ohlcv_rows",
                    )
                    attempts.append(
                        {
                            "exchange": ex,
                            "status": "empty",
                            "error": "empty_ohlcv_rows",
                            "attempt": attempt_idx + 1,
                        }
                    )
                ex_done = True
                break

            if ex_done:
                continue
        except Exception as exc:
            _log_index_api_call(
                market=mk,
                symbol=sym,
                status="exception",
                attempt=1,
                exchange=str(ex),
                detail=repr(exc),
            )
            attempts.append({"exchange": ex, "status": "error", "error": repr(exc)})
            continue

    any_error = any(a.get("status") in {"error", "api_error", "rate_limited"} for a in attempts)
    any_empty = any(a.get("status") in {"empty", "broker_none"} for a in attempts)
    status = "fetch_failed" if any_error else ("empty" if any_empty else "fetch_failed")
    return [], {
        "status": status,
        "market": mk,
        "symbol": sym,
        "attempts": attempts,
        "error": "all_exchanges_failed_or_empty",
    }


def _prev_day_range(selector: Any, market: str, symbol: str, trade_date: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    rows = _fetch_ohlcv_rows(selector, market, symbol)
    if not rows:
        return None, None, None
    hist = [r for r in rows if str(r.get("date", "")) < str(trade_date)]
    if not hist:
        return None, None, None
    prev = hist[-1]
    prev_high = to_float(prev.get("high"))
    prev_low = to_float(prev.get("low"))
    prev_close = to_float(prev.get("close"))
    if prev_high is None and prev_close is not None:
        prev_high = prev_close
    if prev_low is None and prev_close is not None:
        prev_low = prev_close
    return prev_high, prev_low, prev_close


def _market_filter(selector: Any, market: str, index_symbol: str, trade_date: str, ma_days: int) -> Dict[str, Any]:
    rows, diag = _fetch_ohlcv_rows_with_diag(selector, market, index_symbol)
    if str(diag.get("status", "")).lower() == "fetch_failed":
        return {
            "index_symbol": index_symbol,
            "prev_close": None,
            "ma_prev": None,
            "trading_enabled": False,
            "reason": "index_fetch_failed",
            "fetch_diagnostics": diag,
        }
    hist = [r for r in rows if str(r.get("date", "")) < str(trade_date)]
    closes = [float(r["close"]) for r in hist if to_float(r.get("close")) is not None]
    if len(closes) < max(2, ma_days):
        return {
            "index_symbol": index_symbol,
            "prev_close": closes[-1] if closes else None,
            "ma_prev": None,
            "trading_enabled": False,
            "reason": "insufficient_index_bars",
            "bars": len(closes),
            "required_bars": max(2, ma_days),
            "fetch_diagnostics": diag,
        }
    prev_close = float(closes[-1])
    ma_prev = float(mean(closes[-ma_days:]))
    enabled = bool(prev_close > ma_prev)
    reason = "close_above_ma" if enabled else "close_below_or_equal_ma"
    return {
        "index_symbol": index_symbol,
        "prev_close": prev_close,
        "ma_prev": ma_prev,
        "trading_enabled": enabled,
        "reason": reason,
        "bars": len(closes),
        "required_bars": max(2, ma_days),
        "fetch_diagnostics": diag,
    }


def _to_symbol_plan_rows(
    *,
    candidates: List[Stage1Candidate],
    selector: Any,
    market: str,
    trade_date: str,
    breakout_k: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rank, cand in enumerate(candidates, start=1):
        prev_high, prev_low, _ = _prev_day_range(selector, market, cand.code, trade_date)
        today_open = cand.open_price if cand.open_price and cand.open_price > 0 else None
        breakout_price = None
        if today_open is not None and prev_high is not None and prev_low is not None:
            breakout_price = float(today_open + (prev_high - prev_low) * breakout_k)
        rows.append(
            {
                "trade_date": trade_date,
                "market": market,
                "symbol": cand.code,
                "name": cand.name or "",
                "candidate_rank": rank,
                "prev_high": prev_high,
                "prev_low": prev_low,
                "today_open": today_open,
                "breakout_price": breakout_price,
                "entered_today": False,
                "last_price": cand.current_price,
                "is_candidate": True,
            }
        )
    return rows


def precompute_market(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    market: str,
    trade_date: str,
    universe_file: Optional[str] = None,
) -> Dict[str, Any]:
    mk = str(market).upper()
    market_slug = str(mk).lower()
    session_root = cfg.out_dir / market_slug / trade_date
    results_path = session_root / "results" / f"{market_slug}_daily_{trade_date}.json"
    analytics_dir = session_root / "analytics"
    started_at = _now_iso()
    universe_source = "file" if universe_file else "objective"

    status = "OK"
    detail: Dict[str, Any] = {"market": mk}
    rows: List[Dict[str, Any]] = []
    candidate_cache_path = final_candidate_cache_path(cfg, mk, trade_date)
    filter_payload: Dict[str, Any]
    try:
        selector = make_selector(
            cfg=cfg,
            market=mk,
            universe_file=universe_file,
            output_json_path=str(results_path),
            analytics_dir=str(analytics_dir),
        )
        codes, names = selector.load_universe()
        stage1 = selector.build_stage1_candidates(codes, names)
        max_candidates = int(cfg.candidates_max_kr if mk == "KR" else cfg.candidates_max_us)
        stage1 = list(stage1[:max_candidates])
        rows = _to_symbol_plan_rows(
            candidates=stage1,
            selector=selector,
            market=mk,
            trade_date=trade_date,
            breakout_k=cfg.k,
        )
        _write_final_candidate_cache(candidate_cache_path, rows)
        storage.clear_candidate_flags(trade_date, mk)
        storage.upsert_symbol_plan_rows(rows)

        index_symbol = cfg.market_filter_symbol_kr if mk == "KR" else cfg.market_filter_symbol_us
        filter_payload = _market_filter(
            selector=selector,
            market=mk,
            index_symbol=index_symbol,
            trade_date=trade_date,
            ma_days=cfg.market_filter_ma_days,
        )
        storage.upsert_market_filter(
            trade_date=trade_date,
            market=mk,
            index_symbol=str(filter_payload["index_symbol"]),
            prev_close=filter_payload["prev_close"],
            ma20_prev=filter_payload["ma_prev"],
            trading_enabled=bool(filter_payload["trading_enabled"]),
            reason=str(filter_payload["reason"]),
            computed_at=_now_iso(),
        )
        detail.update(
            {
                "universe_count": len(codes),
                "candidate_count": len(rows),
                "candidate_cache_path": str(candidate_cache_path),
                "candidate_cache_count": int(final_candidate_cache_count(candidate_cache_path)),
                "candidate_required_count": int(expected_candidate_count(cfg, mk)),
                "index_filter": filter_payload,
            }
        )
    except Exception as exc:
        status = "ERROR"
        detail["error"] = repr(exc)
        try:
            candidate_cache_path.unlink(missing_ok=True)
        except Exception:
            pass
        rows = []

    finished_at = _now_iso()
    storage.add_premarket_log(
        trade_date=trade_date,
        market=mk,
        started_at=started_at,
        finished_at=finished_at,
        universe_source=universe_source,
        candidate_count=len(rows),
        status=status,
        detail=detail,
    )
    return {
        "market": mk,
        "status": status,
        "candidate_count": len(rows),
        "detail": detail,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def precompute_all_markets(
    *,
    cfg: TraderConfig,
    storage: TraderStorage,
    trade_date: str,
) -> Dict[str, Any]:
    markets = [m for m in ("KR", "US") if cfg.market_enabled(m)]
    results = []
    for market in markets:
        results.append(precompute_market(cfg=cfg, storage=storage, market=market, trade_date=trade_date))
    return {
        "trade_date": trade_date,
        "markets": markets,
        "results": results,
    }
