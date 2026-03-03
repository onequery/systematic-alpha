from __future__ import annotations

from datetime import datetime
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
    rows = _fetch_ohlcv_rows(selector, market, index_symbol)
    hist = [r for r in rows if str(r.get("date", "")) < str(trade_date)]
    closes = [float(r["close"]) for r in hist if to_float(r.get("close")) is not None]
    if len(closes) < max(2, ma_days):
        return {
            "index_symbol": index_symbol,
            "prev_close": closes[-1] if closes else None,
            "ma_prev": None,
            "trading_enabled": False,
            "reason": "insufficient_index_bars",
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
    started_at = _now_iso()
    universe_source = "file" if universe_file else "objective"

    status = "OK"
    detail: Dict[str, Any] = {"market": mk}
    rows: List[Dict[str, Any]] = []
    filter_payload: Dict[str, Any]
    try:
        selector = make_selector(cfg=cfg, market=mk, universe_file=universe_file)
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
                "index_filter": filter_payload,
            }
        )
    except Exception as exc:
        status = "ERROR"
        detail["error"] = repr(exc)
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
