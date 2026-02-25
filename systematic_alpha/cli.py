from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from systematic_alpha.analytics import persist_run_analytics
from systematic_alpha.helpers import fmt
from systematic_alpha.models import (
    FinalSelection,
    RealtimeQuality,
    RealtimeStats,
    Stage1Candidate,
    StrategyConfig,
)
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.selector import DayTradingSelector
from systematic_alpha.selector_us import USDayTradingSelector

REPORT_FIELDS = [
    "market",
    "selection_datetime",
    "selection_date",
    "code",
    "name",
    "rank",
    "recommendation_score",
    "score",
    "max_score",
    "entry_price",
    "selection_close",
    "next_open",
    "next_open_date",
    "intraday_return_pct",
    "overnight_return_pct",
    "total_return_to_next_open_pct",
    "status",
    "last_updated_at",
]


def recommendation_score(item: FinalSelection) -> float:
    if item.max_score <= 0:
        return 0.0
    return (item.score / item.max_score) * 100.0


def log(message: str) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sort_by_recommendation(items: List[FinalSelection]) -> List[FinalSelection]:
    return sorted(
        items,
        key=lambda item: (
            recommendation_score(item),
            item.score,
            item.metrics.get("strength_avg") or 0.0,
            item.metrics.get("volume_ratio") or 0.0,
            item.metrics.get("bid_ask_avg") or 0.0,
            item.metrics.get("prev_day_turnover") or 0.0,
        ),
        reverse=True,
    )


def to_ranked_payload(items: List[FinalSelection]) -> List[dict]:
    payload = []
    for idx, item in enumerate(items, start=1):
        row = asdict(item)
        row["rank"] = idx
        row["recommendation_score"] = round(recommendation_score(item), 4)
        payload.append(row)
    return payload


def summarize_stage1_scan(selector: Any) -> Dict[str, Any]:
    rows = list(getattr(selector, "last_stage1_scan", []) or [])
    if not rows:
        return {}

    passed_count = 0
    reason_counts: Dict[str, int] = {}
    for row in rows:
        if bool(row.get("passed_stage1")):
            passed_count += 1
        reason = str(row.get("skip_reason") or "").strip()
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    sorted_reasons = dict(sorted(reason_counts.items(), key=lambda item: item[1], reverse=True))
    return {
        "scanned": len(rows),
        "passed": passed_count,
        "skip_reason_counts": sorted_reasons,
    }


def selector_api_diagnostics(selector: Any) -> Dict[str, Any]:
    if not hasattr(selector, "get_api_diagnostics"):
        return {}
    try:
        diag = selector.get_api_diagnostics()  # type: ignore[attr-defined]
        if isinstance(diag, dict):
            return diag
    except Exception:
        pass
    return {}


def _stage1_no_price_failfast_enabled() -> bool:
    raw = str(os.getenv("AGENT_LAB_STAGE1_NO_PRICE_FAILFAST", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _detect_no_price_snapshot_burst(stage1_scan_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not stage1_scan_summary:
        return {"burst": False, "ratio": 0.0, "scanned": 0, "no_price_snapshot": 0}
    scanned = int(stage1_scan_summary.get("scanned", 0) or 0)
    reasons = stage1_scan_summary.get("skip_reason_counts", {}) or {}
    no_price = int(reasons.get("no_price_snapshot", 0) or 0) if isinstance(reasons, dict) else 0
    ratio = (float(no_price) / float(scanned)) if scanned > 0 else 0.0
    min_scanned = max(10, int(float(os.getenv("AGENT_LAB_STAGE1_NO_PRICE_FAILFAST_MIN_SCANNED", "40") or 40)))
    min_ratio = float(os.getenv("AGENT_LAB_STAGE1_NO_PRICE_FAILFAST_RATIO", "0.8") or 0.8)
    burst = _stage1_no_price_failfast_enabled() and scanned >= min_scanned and ratio >= min_ratio
    return {
        "burst": bool(burst),
        "ratio": float(ratio),
        "scanned": int(scanned),
        "no_price_snapshot": int(no_price),
        "min_scanned": int(min_scanned),
        "min_ratio": float(min_ratio),
    }


def save_json_output(
    config: StrategyConfig,
    decision_at: datetime,
    realtime_quality: RealtimeQuality,
    final: List[FinalSelection],
    ranked: List[FinalSelection],
    invalid_reason: Optional[str] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> None:
    if not config.output_json_path:
        return
    out_path = Path(config.output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "market": config.market,
                "exchange": config.us_exchange if config.market == "US" else "KRX",
                "generated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "decision_at": decision_at.isoformat(),
                "signal_valid": invalid_reason is None,
                "invalid_reason": invalid_reason,
                "realtime_ready": realtime_quality.realtime_ready,
                "realtime_quality": asdict(realtime_quality),
                "sorted_by": "recommendation_score_desc",
                "final": to_ranked_payload(final),
                "all_ranked": to_ranked_payload(ranked),
                "debug": debug or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")


def print_final_table(final: List[FinalSelection]) -> None:
    print("")
    print("Final Picks")
    if not final:
        print("(no picks)")
        return
    print("rank code    rec%   score  change%  gap%    volRatio strength bid/ask  vwapOK")
    for idx, item in enumerate(final, start=1):
        vol_ratio = item.metrics.get("volume_ratio")
        strength = item.metrics.get("strength_avg")
        bid_ask = item.metrics.get("bid_ask_avg")
        vwap = item.metrics.get("vwap")
        latest = item.metrics.get("latest_price")
        vwap_ok = latest is not None and vwap is not None and latest >= vwap
        print(
            f"{idx:>4} {item.code:<6} {recommendation_score(item):>6.2f} "
            f"{item.score:>2}/{item.max_score:<3} "
            f"{fmt(item.metrics.get('current_change_pct')):>7} "
            f"{fmt(item.metrics.get('gap_pct')):>7} "
            f"{fmt(vol_ratio, 3):>8} "
            f"{fmt(strength):>8} "
            f"{fmt(bid_ask, 3):>7} "
            f"{'Y' if vwap_ok else 'N':>6}"
        )


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_csv_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _read_report_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_report_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REPORT_FIELDS})


def _resolve_report_status(metrics: Dict[str, Optional[float | str]]) -> str:
    if metrics.get("next_open") is not None:
        return "complete"
    if metrics.get("selection_close") is not None:
        return "day_close_ready"
    return "pending"


def _discover_market_report_paths(report_path: Optional[str], market: str) -> List[Path]:
    if not report_path:
        return []
    current_path = Path(report_path)
    market_tag = market.strip().lower()
    if not market_tag:
        return [current_path]

    # Expected layout: out/{kr|us}/YYYYMMDD/selection_overnight_report.csv
    # Legacy layout: out/YYYYMMDD/{kr|us}/selection_overnight_report.csv
    # If layout differs, fallback to current path only.
    try:
        out_root = current_path.parent.parent.parent
        if out_root.exists():
            discovered = sorted(
                set(
                    list(out_root.glob(f"{market_tag}/*/selection_overnight_report.csv"))
                    + list(out_root.glob(f"*/{market_tag}/selection_overnight_report.csv"))
                )
            )
            if current_path not in discovered:
                discovered.append(current_path)
            return sorted(set(discovered))
    except Exception:
        pass
    return [current_path]


def update_pending_overnight_reports(selector: Any, report_path: Optional[str], market: str) -> None:
    report_paths = _discover_market_report_paths(report_path, market)
    if not report_paths:
        return

    total_changed_rows = 0
    updated_files = 0
    now_iso = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()

    for path in report_paths:
        if not path.exists():
            continue
        rows = _read_report_rows(path)
        if not rows:
            continue

        changed = False
        updated_count = 0
        for row in rows:
            if row.get("status") == "complete":
                continue
            code = row.get("code", "").strip()
            selection_date = row.get("selection_date", "").strip()
            entry_price = _parse_float(row.get("entry_price", ""))
            if not code or not selection_date:
                continue

            metrics = selector.build_overnight_report_metrics(code, selection_date, entry_price)
            new_status = _resolve_report_status(metrics)

            before = (
                row.get("selection_close", ""),
                row.get("next_open", ""),
                row.get("intraday_return_pct", ""),
                row.get("overnight_return_pct", ""),
                row.get("total_return_to_next_open_pct", ""),
                row.get("status", ""),
            )

            row["selection_close"] = _fmt_csv_float(metrics.get("selection_close"))  # type: ignore[arg-type]
            row["next_open"] = _fmt_csv_float(metrics.get("next_open"))  # type: ignore[arg-type]
            row["next_open_date"] = str(metrics.get("next_open_date") or "")
            row["intraday_return_pct"] = _fmt_csv_float(metrics.get("intraday_return_pct"))  # type: ignore[arg-type]
            row["overnight_return_pct"] = _fmt_csv_float(metrics.get("overnight_return_pct"))  # type: ignore[arg-type]
            row["total_return_to_next_open_pct"] = _fmt_csv_float(
                metrics.get("total_return_to_next_open_pct")  # type: ignore[arg-type]
            )
            row["status"] = new_status
            row["last_updated_at"] = now_iso

            after = (
                row.get("selection_close", ""),
                row.get("next_open", ""),
                row.get("intraday_return_pct", ""),
                row.get("overnight_return_pct", ""),
                row.get("total_return_to_next_open_pct", ""),
                row.get("status", ""),
            )
            if before != after:
                changed = True
                updated_count += 1

        if changed:
            _write_report_rows(path, rows)
            total_changed_rows += updated_count
            updated_files += 1

    if total_changed_rows > 0:
        log(
            f"[overnight-report] updated pending rows: {total_changed_rows} "
            f"(files={updated_files})"
        )


def append_selection_report_rows(
    selector: Any,
    report_path: Optional[str],
    final: List[FinalSelection],
    decision_at: datetime,
) -> None:
    if not report_path or not final:
        return

    path = Path(report_path)
    rows = _read_report_rows(path)
    selection_date = decision_at.strftime("%Y%m%d")
    now_iso = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()

    for idx, item in enumerate(final, start=1):
        entry_price = item.metrics.get("latest_price")
        metrics = selector.build_overnight_report_metrics(item.code, selection_date, entry_price)
        status = _resolve_report_status(metrics)
        rows.append(
            {
                "market": selector.config.market,
                "selection_datetime": decision_at.isoformat(),
                "selection_date": selection_date,
                "code": item.code,
                "name": item.name,
                "rank": str(idx),
                "recommendation_score": f"{recommendation_score(item):.4f}",
                "score": str(item.score),
                "max_score": str(item.max_score),
                "entry_price": _fmt_csv_float(entry_price),
                "selection_close": _fmt_csv_float(metrics.get("selection_close")),  # type: ignore[arg-type]
                "next_open": _fmt_csv_float(metrics.get("next_open")),  # type: ignore[arg-type]
                "next_open_date": str(metrics.get("next_open_date") or ""),
                "intraday_return_pct": _fmt_csv_float(metrics.get("intraday_return_pct")),  # type: ignore[arg-type]
                "overnight_return_pct": _fmt_csv_float(metrics.get("overnight_return_pct")),  # type: ignore[arg-type]
                "total_return_to_next_open_pct": _fmt_csv_float(
                    metrics.get("total_return_to_next_open_pct")  # type: ignore[arg-type]
                ),
                "status": status,
                "last_updated_at": now_iso,
            }
        )

    _write_report_rows(path, rows)
    log(f"[overnight-report] appended rows: {len(final)} -> {path}")


def build_assume_open_stage1_candidates(
    selector: Any,
    codes: List[str],
    names: Dict[str, str],
    limit: int,
) -> List[Stage1Candidate]:
    generated: List[Stage1Candidate] = []
    target = max(1, limit)
    for code in codes:
        synthetic_seed = float(100.0 + len(generated))
        try:
            snap = selector.fetch_price_snapshot(code, use_cache=False)
            prev = selector.fetch_prev_day_stats(code)
        except Exception:
            snap = None
            prev = None

        if not snap:
            name = names.get(code, "") or code
            generated.append(
                Stage1Candidate(
                    code=code,
                    name=name,
                    current_price=synthetic_seed,
                    open_price=synthetic_seed,
                    current_change_pct=0.0,
                    gap_pct=0.0,
                    prev_close=synthetic_seed,
                    prev_day_volume=1_000_000.0,
                    prev_day_turnover=synthetic_seed * 1_000_000.0,
                )
            )
            if len(generated) >= target:
                break
            continue

        price = snap.get("price")
        if price is None or price <= 0:
            continue

        open_price = snap.get("open")
        if open_price is None or open_price <= 0:
            open_price = price

        if prev is not None and prev.prev_close > 0:
            prev_close = float(prev.prev_close)
            prev_day_volume = float(prev.prev_volume)
            prev_day_turnover = float(prev.prev_turnover)
        else:
            # Off-market synthetic fallback for test mode when prev-day bars are unavailable.
            prev_close = float(open_price if open_price > 0 else price)
            prev_day_volume = float(max(1.0, snap.get("acml_vol") or 1.0))
            prev_day_turnover = float(max(prev_close * prev_day_volume, prev_close))

        change_pct = snap.get("change_pct")
        if change_pct is None:
            change_pct = ((price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        gap_pct = ((open_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        name = names.get(code, "") or str(snap.get("name") or "")

        generated.append(
            Stage1Candidate(
                code=code,
                name=name,
                current_price=float(price),
                open_price=float(open_price),
                current_change_pct=float(change_pct),
                gap_pct=float(gap_pct),
                prev_close=prev_close,
                prev_day_volume=prev_day_volume,
                prev_day_turnover=prev_day_turnover,
            )
        )
        if len(generated) >= target:
            break
    return generated


def inject_assume_open_realtime(
    selector: Any,
    config: StrategyConfig,
    candidates: List[Stage1Candidate],
    stats: Dict[str, RealtimeStats],
) -> RealtimeQuality:
    if hasattr(selector, "_orderbook_available"):
        setattr(selector, "_orderbook_available", True)

    strength_floor = max(config.min_strength + 5.0, 120.0)
    bid_ask_floor = max(config.min_bid_ask_ratio + 0.1, 1.25)
    strength_samples = max(3, config.min_strength_samples)
    bid_ask_samples = max(3, config.min_bid_ask_samples)

    for candidate in candidates:
        realtime = stats.get(candidate.code, RealtimeStats())
        px = candidate.current_price if candidate.current_price > 0 else max(candidate.open_price, 1.0)

        realtime.got_execution = True
        realtime.execution_ticks = max(realtime.execution_ticks, config.min_exec_ticks, 30)
        realtime.got_orderbook = True
        realtime.orderbook_ticks = max(realtime.orderbook_ticks, config.min_orderbook_ticks, 30)

        if len(realtime.strength_values) < strength_samples:
            realtime.strength_values = [strength_floor] * strength_samples
        if len(realtime.bid_ask_ratios) < bid_ask_samples:
            realtime.bid_ask_ratios = [bid_ask_floor] * bid_ask_samples

        realtime.cum_trade_volume = max(
            realtime.cum_trade_volume,
            config.min_realtime_cum_volume + float(realtime.execution_ticks),
        )
        realtime.cum_trade_value = max(realtime.cum_trade_value, px * realtime.cum_trade_volume)
        realtime.latest_price = realtime.latest_price if realtime.latest_price is not None else px

        min_acml_volume = max(1.0, candidate.prev_day_volume * max(config.min_vol_ratio, 0.1) * 1.2)
        current_acml = realtime.latest_acml_volume or 0.0
        realtime.latest_acml_volume = max(current_acml, min_acml_volume)

        if realtime.first_reported_low is None:
            realtime.first_reported_low = candidate.open_price if candidate.open_price > 0 else px
        realtime.low_broken_after_start = False
        stats[candidate.code] = realtime

    total = len(candidates)
    return RealtimeQuality(
        realtime_ready=total > 0,
        quality_ok=True,
        coverage_ratio=1.0 if total > 0 else 0.0,
        eligible_count=total,
        total_count=total,
        min_exec_ticks=config.min_exec_ticks,
        min_orderbook_ticks=config.min_orderbook_ticks,
        min_realtime_cum_volume=config.min_realtime_cum_volume,
    )


def run(config: StrategyConfig) -> None:
    total_started = perf_counter()
    run_started_at = datetime.now(ZoneInfo("Asia/Seoul"))
    log(
        "Run config: "
        f"market={config.market}, "
        f"exchange={config.us_exchange if config.market == 'US' else 'KRX'}, "
        f"collect={config.collect_seconds}s, "
        f"final_picks={config.final_picks}, pre_candidates={config.pre_candidates}, "
        f"max_symbols_scan={config.max_symbols_scan}, "
        f"long_only={config.long_only}, "
        f"test_assume_open={config.test_assume_open}, "
        f"min_coverage={config.min_realtime_coverage_ratio:.2f}, "
        f"invalidate_on_low_coverage={config.invalidate_on_low_coverage}, "
        f"stage1_log_interval={config.stage1_log_interval}, "
        f"realtime_log_interval={config.realtime_log_interval}s"
    )
    mojito_module = import_mojito_module()
    if config.market == "US":
        selector = USDayTradingSelector(mojito_module, config)
    else:
        selector = DayTradingSelector(mojito_module, config)

    codes: List[str] = []
    stage1_initial: List[Stage1Candidate] = []
    stage1: List[Stage1Candidate] = []
    decision_candidates: List[Stage1Candidate] = []
    ranked: List[FinalSelection] = []
    final: List[FinalSelection] = []
    realtime_stats: Dict[str, RealtimeStats] = {}
    fallback_added_count = 0
    stage1_scan_summary: Dict[str, Any] = {}
    timings_sec: Dict[str, float] = {
        "load_universe_sec": 0.0,
        "stage1_sec": 0.0,
        "fallback_sec": 0.0,
        "realtime_sec": 0.0,
        "refresh_sec": 0.0,
        "scoring_sec": 0.0,
    }
    realtime_quality = RealtimeQuality(
        realtime_ready=False,
        quality_ok=True,
        coverage_ratio=0.0,
        eligible_count=0,
        total_count=0,
        min_exec_ticks=config.min_exec_ticks,
        min_orderbook_ticks=config.min_orderbook_ticks,
        min_realtime_cum_volume=config.min_realtime_cum_volume,
    )

    def build_debug_payload() -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        summary = summarize_stage1_scan(selector)
        if summary:
            payload["stage1_scan_summary"] = summary
        api_diag = selector_api_diagnostics(selector)
        if api_diag:
            payload["api_diagnostics"] = api_diag
        return payload

    def persist_analytics_snapshot(decision_at: datetime, invalid_reason: Optional[str]) -> None:
        if not config.enable_analytics_log:
            return

        run_finished_at = datetime.now(ZoneInfo("Asia/Seoul"))
        timings_payload = dict(timings_sec)
        timings_payload["total_sec"] = perf_counter() - total_started
        try:
            paths = persist_run_analytics(
                config=config,
                selector=selector,
                run_started_at=run_started_at,
                decision_at=decision_at,
                run_finished_at=run_finished_at,
                universe_size=len(codes),
                stage1_initial=stage1_initial,
                stage1_final=stage1,
                decision_candidates=decision_candidates,
                realtime_stats=realtime_stats,
                realtime_quality=realtime_quality,
                ranked=ranked,
                final=final,
                fallback_added_count=fallback_added_count,
                invalid_reason=invalid_reason,
                timings_sec=timings_payload,
            )
            if paths:
                log(f"[analytics] run bundle saved: {paths['run_bundle']}")
                log(
                    "[analytics] tables updated: "
                    f"summary={paths['run_summary']}, "
                    f"stage1={paths['stage1_scan']}, "
                    f"ranked={paths['ranked_symbols']}"
                )
        except Exception as exc:
            log(f"[analytics] persist failed: {exc}")

    if config.skip_overnight_report_update:
        log("[overnight-report] pending-row update skipped by config.")
    else:
        update_pending_overnight_reports(selector, config.overnight_report_path, config.market)

    stage_started = perf_counter()
    log("[1/4] Loading universe...")
    codes, names = selector.load_universe()
    if not codes:
        raise RuntimeError("No symbols loaded. Provide --universe-file or check fetch_symbols().")
    timings_sec["load_universe_sec"] = perf_counter() - stage_started
    log(f"Universe size: {len(codes)} (elapsed {timings_sec['load_universe_sec']:.1f}s)")

    stage_started = perf_counter()
    log("[2/4] Stage1 filtering (change/gap/prev turnover)...")
    stage1 = selector.build_stage1_candidates(codes, names)
    stage1_initial = list(stage1)
    stage1_scan_summary = summarize_stage1_scan(selector)
    timings_sec["stage1_sec"] = perf_counter() - stage_started
    log(f"Stage1 candidates: {len(stage1)} (elapsed {timings_sec['stage1_sec']:.1f}s)")
    if stage1_scan_summary:
        top_skip_counts = list(stage1_scan_summary.get("skip_reason_counts", {}).items())[:3]
        top_skip_text = ", ".join(f"{k}={v}" for k, v in top_skip_counts) if top_skip_counts else "none"
        log(
            "[stage1] scan summary: "
            f"scanned={stage1_scan_summary.get('scanned', 0)}, "
            f"passed={stage1_scan_summary.get('passed', 0)}, "
            f"top_skips={top_skip_text}"
        )
    no_price_burst = _detect_no_price_snapshot_burst(stage1_scan_summary)
    if no_price_burst.get("burst", False):
        log(
            "[stage1] fail-fast trigger: "
            f"no_price_snapshot={no_price_burst['no_price_snapshot']}/{no_price_burst['scanned']} "
            f"({no_price_burst['ratio']:.1%}) >= threshold({no_price_burst['min_ratio']:.1%})"
        )

    if len(stage1) < config.final_picks:
        if no_price_burst.get("burst", False):
            log(
                "[fallback] skipped due to stage1 no_price_snapshot burst "
                "(price API unavailable suspected)."
            )
        else:
            fallback_started = perf_counter()
            needed = config.final_picks - len(stage1)
            log(
                f"[fallback] stage1 candidates ({len(stage1)}) < final picks ({config.final_picks}). applying relaxed fallback rules...",
            )
            fallback_candidates = selector.build_fallback_candidates(
                codes=codes,
                names=names,
                exclude_codes={item.code for item in stage1},
                needed=needed,
            )
            if fallback_candidates:
                fallback_added_count = len(fallback_candidates)
                stage1.extend(fallback_candidates)
                log(f"[fallback] added {len(fallback_candidates)} candidates. total stage1={len(stage1)}")
            else:
                log("[fallback] no additional candidates found.")
            timings_sec["fallback_sec"] = perf_counter() - fallback_started

    if not stage1 and config.test_assume_open:
        force_limit = max(config.pre_candidates, config.final_picks * 2)
        log(
            "[test-assume-open] stage1 is empty. building synthetic stage1 from current snapshots for off-market testing..."
        )
        stage1 = build_assume_open_stage1_candidates(
            selector=selector,
            codes=codes,
            names=names,
            limit=force_limit,
        )
        log(f"[test-assume-open] synthetic stage1 candidates: {len(stage1)}")

    if not stage1:
        log("No candidates passed stage1 thresholds.")
        api_diag = selector_api_diagnostics(selector)
        if api_diag:
            log(f"[api-diag] {api_diag}")
        invalid_reason = "price_snapshot_unavailable" if no_price_burst.get("burst", False) else "no_stage1_candidates"
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        no_realtime = RealtimeQuality(
            realtime_ready=False,
            quality_ok=True,
            coverage_ratio=0.0,
            eligible_count=0,
            total_count=0,
            min_exec_ticks=config.min_exec_ticks,
            min_orderbook_ticks=config.min_orderbook_ticks,
            min_realtime_cum_volume=config.min_realtime_cum_volume,
        )
        save_json_output(
            config=config,
            decision_at=now,
            realtime_quality=no_realtime,
            final=[],
            ranked=[],
            invalid_reason=invalid_reason,
            debug=build_debug_payload(),
        )
        persist_analytics_snapshot(decision_at=now, invalid_reason=invalid_reason)
        log(f"Run finished (total elapsed {perf_counter() - total_started:.1f}s)")
        return

    stage_started = perf_counter()
    log("[3/4] Collecting realtime data (strength/VWAP/orderbook)...")
    target_codes = [item.code for item in stage1]
    realtime_stats, realtime_quality = selector.collect_realtime(target_codes)
    timings_sec["realtime_sec"] = perf_counter() - stage_started
    log(
        "[realtime] quality summary: "
        f"ready={realtime_quality.realtime_ready}, "
        f"coverage={realtime_quality.coverage_ratio:.3f}, "
        f"eligible={realtime_quality.eligible_count}/{realtime_quality.total_count}, "
        f"quality_ok={realtime_quality.quality_ok}"
    )
    log(f"[3/4] Realtime collection done (elapsed {timings_sec['realtime_sec']:.1f}s)")

    if config.test_assume_open and (not realtime_quality.realtime_ready or not realtime_quality.quality_ok):
        log(
            "[test-assume-open] realtime data is insufficient/off-market. injecting synthetic realtime metrics to force full pipeline test."
        )
        realtime_quality = inject_assume_open_realtime(
            selector=selector,
            config=config,
            candidates=stage1,
            stats=realtime_stats,
        )
        log(
            "[test-assume-open] quality summary overridden: "
            f"ready={realtime_quality.realtime_ready}, "
            f"coverage={realtime_quality.coverage_ratio:.3f}, "
            f"eligible={realtime_quality.eligible_count}/{realtime_quality.total_count}, "
            f"quality_ok={realtime_quality.quality_ok}"
        )

    if (
        config.collect_seconds > 0
        and config.invalidate_on_low_coverage
        and not config.test_assume_open
        and not realtime_quality.quality_ok
    ):
        invalid_reason = (
            "realtime_coverage_too_low:"
            f"{realtime_quality.coverage_ratio:.3f}<{config.min_realtime_coverage_ratio:.3f}"
        )
        log(f"[invalid] {invalid_reason} -> no picks for today.")
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        print_final_table([])
        save_json_output(
            config=config,
            decision_at=now,
            realtime_quality=realtime_quality,
            final=[],
            ranked=[],
            invalid_reason=invalid_reason,
            debug=build_debug_payload(),
        )
        persist_analytics_snapshot(decision_at=now, invalid_reason=invalid_reason)
        print("\nTop codes: (none)")
        log(f"Run finished (total elapsed {perf_counter() - total_started:.1f}s)")
        return

    stage_started = perf_counter()
    log("[3.5/4] Refreshing stage1 snapshots at decision time...")
    decision_candidates = selector.refresh_candidates_for_decision(stage1)
    timings_sec["refresh_sec"] = perf_counter() - stage_started
    log(f"[3.5/4] Snapshot refresh done (elapsed {timings_sec['refresh_sec']:.1f}s)")

    stage_started = perf_counter()
    log("[4/4] Final scoring...")
    ranked = sort_by_recommendation(
        selector.evaluate(decision_candidates, realtime_stats, realtime_quality.realtime_ready)
    )
    timings_sec["scoring_sec"] = perf_counter() - stage_started
    log(f"Scoring complete (elapsed {timings_sec['scoring_sec']:.1f}s). Ranked={len(ranked)}")

    passed = [item for item in ranked if item.passed]
    final = passed[: config.final_picks]
    if len(final) < config.final_picks:
        needed = config.final_picks - len(final)
        final.extend([item for item in ranked if item not in final][:needed])

    decision_at = datetime.now(ZoneInfo("Asia/Seoul"))
    print_final_table(final)
    save_json_output(
        config=config,
        decision_at=decision_at,
        realtime_quality=realtime_quality,
        final=final,
        ranked=ranked,
        debug=build_debug_payload(),
    )
    if config.skip_overnight_report_append:
        log("[overnight-report] append skipped by config.")
    else:
        append_selection_report_rows(selector, config.overnight_report_path, final, decision_at)
    persist_analytics_snapshot(decision_at=decision_at, invalid_reason=None)
    print("\nTop codes:", ", ".join(item.code for item in final) if final else "(none)")
    log(f"Run finished (total elapsed {perf_counter() - total_started:.1f}s)")
