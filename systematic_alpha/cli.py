from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

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


def save_json_output(
    config: StrategyConfig,
    decision_at: datetime,
    realtime_quality: RealtimeQuality,
    final: List[FinalSelection],
    ranked: List[FinalSelection],
    invalid_reason: Optional[str] = None,
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


def update_pending_overnight_report(selector: DayTradingSelector, report_path: Optional[str]) -> None:
    if not report_path:
        return
    path = Path(report_path)
    if not path.exists():
        return

    rows = _read_report_rows(path)
    if not rows:
        return

    changed = False
    updated_count = 0
    now_iso = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()

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
        log(f"[overnight-report] updated pending rows: {updated_count}")


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
        try:
            snap = selector.fetch_price_snapshot(code, use_cache=False)
            prev = selector.fetch_prev_day_stats(code)
        except Exception:
            continue
        if not snap or prev is None or prev.prev_close <= 0:
            continue

        price = snap.get("price")
        if price is None or price <= 0:
            continue

        open_price = snap.get("open")
        if open_price is None or open_price <= 0:
            open_price = price

        change_pct = snap.get("change_pct")
        if change_pct is None:
            change_pct = ((price - prev.prev_close) / prev.prev_close) * 100.0
        gap_pct = ((open_price - prev.prev_close) / prev.prev_close) * 100.0
        name = names.get(code, "") or str(snap.get("name") or "")

        generated.append(
            Stage1Candidate(
                code=code,
                name=name,
                current_price=float(price),
                open_price=float(open_price),
                current_change_pct=float(change_pct),
                gap_pct=float(gap_pct),
                prev_close=float(prev.prev_close),
                prev_day_volume=float(prev.prev_volume),
                prev_day_turnover=float(prev.prev_turnover),
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

    update_pending_overnight_report(selector, config.overnight_report_path)

    stage_started = perf_counter()
    log("[1/4] Loading universe...")
    codes, names = selector.load_universe()
    if not codes:
        raise RuntimeError("No symbols loaded. Provide --universe-file or check fetch_symbols().")
    log(f"Universe size: {len(codes)} (elapsed {perf_counter() - stage_started:.1f}s)")

    stage_started = perf_counter()
    log("[2/4] Stage1 filtering (change/gap/prev turnover)...")
    stage1 = selector.build_stage1_candidates(codes, names)
    log(f"Stage1 candidates: {len(stage1)} (elapsed {perf_counter() - stage_started:.1f}s)")

    if len(stage1) < config.final_picks:
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
            stage1.extend(fallback_candidates)
            log(f"[fallback] added {len(fallback_candidates)} candidates. total stage1={len(stage1)}")
        else:
            log("[fallback] no additional candidates found.")

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
            invalid_reason="no_stage1_candidates",
        )
        return

    stage_started = perf_counter()
    log("[3/4] Collecting realtime data (strength/VWAP/orderbook)...")
    target_codes = [item.code for item in stage1]
    realtime_stats, realtime_quality = selector.collect_realtime(target_codes)
    log(
        "[realtime] quality summary: "
        f"ready={realtime_quality.realtime_ready}, "
        f"coverage={realtime_quality.coverage_ratio:.3f}, "
        f"eligible={realtime_quality.eligible_count}/{realtime_quality.total_count}, "
        f"quality_ok={realtime_quality.quality_ok}"
    )
    log(f"[3/4] Realtime collection done (elapsed {perf_counter() - stage_started:.1f}s)")

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
        )
        print("\nTop codes: (none)")
        log(f"Run finished (total elapsed {perf_counter() - total_started:.1f}s)")
        return

    stage_started = perf_counter()
    log("[3.5/4] Refreshing stage1 snapshots at decision time...")
    decision_candidates = selector.refresh_candidates_for_decision(stage1)
    log(f"[3.5/4] Snapshot refresh done (elapsed {perf_counter() - stage_started:.1f}s)")

    stage_started = perf_counter()
    log("[4/4] Final scoring...")
    ranked = sort_by_recommendation(
        selector.evaluate(decision_candidates, realtime_stats, realtime_quality.realtime_ready)
    )
    log(f"Scoring complete (elapsed {perf_counter() - stage_started:.1f}s). Ranked={len(ranked)}")

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
    )
    append_selection_report_rows(selector, config.overnight_report_path, final, decision_at)
    print("\nTop codes:", ", ".join(item.code for item in final) if final else "(none)")
    log(f"Run finished (total elapsed {perf_counter() - total_started:.1f}s)")
