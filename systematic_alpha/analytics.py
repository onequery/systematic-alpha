from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from systematic_alpha.models import (
    FinalSelection,
    RealtimeQuality,
    RealtimeStats,
    Stage1Candidate,
    StrategyConfig,
)

RUN_SUMMARY_FIELDS = [
    "run_id",
    "market",
    "exchange",
    "run_started_at",
    "decision_at",
    "run_finished_at",
    "signal_valid",
    "invalid_reason",
    "universe_size",
    "stage1_initial_count",
    "fallback_added_count",
    "stage1_final_count",
    "decision_candidate_count",
    "ranked_count",
    "final_count",
    "realtime_ready",
    "realtime_quality_ok",
    "realtime_coverage_ratio",
    "realtime_eligible_count",
    "realtime_total_count",
    "collect_seconds",
    "final_picks",
    "pre_candidates",
    "max_symbols_scan",
    "long_only",
    "min_change_pct",
    "min_gap_pct",
    "min_prev_turnover",
    "min_strength",
    "min_vol_ratio",
    "min_bid_ask_ratio",
    "min_pass_conditions",
    "min_maintain_ratio",
    "min_realtime_coverage_ratio",
    "load_universe_sec",
    "stage1_sec",
    "fallback_sec",
    "realtime_sec",
    "refresh_sec",
    "scoring_sec",
    "total_sec",
    "final_codes",
]

STAGE1_SCAN_FIELDS = [
    "run_id",
    "market",
    "decision_at",
    "scan_index",
    "code",
    "name",
    "current_price",
    "open_price",
    "change_pct",
    "gap_pct",
    "prev_close",
    "prev_day_volume",
    "prev_day_turnover",
    "pass_change",
    "pass_gap",
    "pass_prev_turnover",
    "passed_stage1",
    "skip_reason",
    "min_change_pct",
    "min_gap_pct",
    "min_prev_turnover",
    "long_only",
]

RANKED_SYMBOL_FIELDS = [
    "run_id",
    "market",
    "decision_at",
    "rank",
    "is_final_pick",
    "selected_slot",
    "code",
    "name",
    "recommendation_score",
    "score",
    "max_score",
    "passed",
    "cond_change_pct",
    "cond_gap_pct",
    "cond_prev_turnover",
    "cond_strength_maintained",
    "cond_volume_ratio",
    "cond_bid_ask_maintained",
    "cond_price_above_vwap",
    "cond_low_not_broken",
    "metric_current_change_pct",
    "metric_gap_pct",
    "metric_prev_day_turnover",
    "metric_prev_day_volume",
    "metric_strength_avg",
    "metric_strength_hit_ratio",
    "metric_bid_ask_avg",
    "metric_bid_ask_hit_ratio",
    "metric_volume_ratio",
    "metric_vwap",
    "metric_latest_price",
    "metric_execution_ticks",
    "metric_orderbook_ticks",
    "metric_realtime_eligible",
    "rt_cum_trade_volume",
    "rt_cum_trade_value",
    "rt_latest_acml_volume",
    "rt_first_reported_low",
    "rt_low_broken_after_start",
    "rt_strength_samples",
    "rt_bid_ask_samples",
]


def _bool_to_int(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "1" if value else "0"


def _recommendation_score(item: FinalSelection) -> float:
    if item.max_score <= 0:
        return 0.0
    return (item.score / item.max_score) * 100.0


def _append_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _serialize_realtime_stats(stats: RealtimeStats) -> Dict[str, Any]:
    return {
        "got_execution": stats.got_execution,
        "got_orderbook": stats.got_orderbook,
        "execution_ticks": stats.execution_ticks,
        "orderbook_ticks": stats.orderbook_ticks,
        "strength_values": stats.strength_values,
        "bid_ask_ratios": stats.bid_ask_ratios,
        "cum_trade_volume": stats.cum_trade_volume,
        "cum_trade_value": stats.cum_trade_value,
        "latest_price": stats.latest_price,
        "latest_acml_volume": stats.latest_acml_volume,
        "first_reported_low": stats.first_reported_low,
        "low_broken_after_start": stats.low_broken_after_start,
    }


def _sanitize_config(config: StrategyConfig) -> Dict[str, Any]:
    return {
        "market": config.market,
        "exchange": config.us_exchange if config.market == "US" else "KRX",
        "mock": config.mock,
        "collect_seconds": config.collect_seconds,
        "max_symbols_scan": config.max_symbols_scan,
        "kr_universe_size": config.kr_universe_size,
        "us_universe_size": config.us_universe_size,
        "pre_candidates": config.pre_candidates,
        "final_picks": config.final_picks,
        "rest_sleep_sec": config.rest_sleep_sec,
        "min_change_pct": config.min_change_pct,
        "min_gap_pct": config.min_gap_pct,
        "min_prev_turnover": config.min_prev_turnover,
        "min_strength": config.min_strength,
        "min_vol_ratio": config.min_vol_ratio,
        "min_bid_ask_ratio": config.min_bid_ask_ratio,
        "min_pass_conditions": config.min_pass_conditions,
        "min_maintain_ratio": config.min_maintain_ratio,
        "min_strength_samples": config.min_strength_samples,
        "min_bid_ask_samples": config.min_bid_ask_samples,
        "long_only": config.long_only,
        "min_exec_ticks": config.min_exec_ticks,
        "min_orderbook_ticks": config.min_orderbook_ticks,
        "min_realtime_cum_volume": config.min_realtime_cum_volume,
        "min_realtime_coverage_ratio": config.min_realtime_coverage_ratio,
        "invalidate_on_low_coverage": config.invalidate_on_low_coverage,
        "stage1_log_interval": config.stage1_log_interval,
        "realtime_log_interval": config.realtime_log_interval,
        "overnight_report_path": config.overnight_report_path,
        "output_json_path": config.output_json_path,
        "analytics_dir": config.analytics_dir,
        "enable_analytics_log": config.enable_analytics_log,
        "test_assume_open": config.test_assume_open,
    }


def _ranked_payload(items: List[FinalSelection]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        row = asdict(item)
        row["rank"] = idx
        row["recommendation_score"] = round(_recommendation_score(item), 6)
        payload.append(row)
    return payload


def persist_run_analytics(
    *,
    config: StrategyConfig,
    selector: Any,
    run_started_at: datetime,
    decision_at: datetime,
    run_finished_at: datetime,
    universe_size: int,
    stage1_initial: List[Stage1Candidate],
    stage1_final: List[Stage1Candidate],
    decision_candidates: List[Stage1Candidate],
    realtime_stats: Dict[str, RealtimeStats],
    realtime_quality: RealtimeQuality,
    ranked: List[FinalSelection],
    final: List[FinalSelection],
    fallback_added_count: int,
    invalid_reason: Optional[str],
    timings_sec: Dict[str, float],
) -> Dict[str, Path]:
    if not config.enable_analytics_log:
        return {}

    base_dir = Path(config.analytics_dir or "./out/analytics")
    decision_day = decision_at.strftime("%Y%m%d")
    all_dir = base_dir / "all"
    daily_dir = base_dir / "daily" / decision_day
    runs_dir = base_dir / "runs" / decision_day
    all_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{config.market.lower()}_{decision_at.strftime('%Y%m%d_%H%M%S_%f')}"
    exchange = config.us_exchange if config.market == "US" else "KRX"
    final_slot = {item.code: idx for idx, item in enumerate(final, start=1)}
    signal_valid = invalid_reason is None

    summary_row = {
        "run_id": run_id,
        "market": config.market,
        "exchange": exchange,
        "run_started_at": run_started_at.isoformat(),
        "decision_at": decision_at.isoformat(),
        "run_finished_at": run_finished_at.isoformat(),
        "signal_valid": _bool_to_int(signal_valid),
        "invalid_reason": invalid_reason or "",
        "universe_size": universe_size,
        "stage1_initial_count": len(stage1_initial),
        "fallback_added_count": fallback_added_count,
        "stage1_final_count": len(stage1_final),
        "decision_candidate_count": len(decision_candidates),
        "ranked_count": len(ranked),
        "final_count": len(final),
        "realtime_ready": _bool_to_int(realtime_quality.realtime_ready),
        "realtime_quality_ok": _bool_to_int(realtime_quality.quality_ok),
        "realtime_coverage_ratio": realtime_quality.coverage_ratio,
        "realtime_eligible_count": realtime_quality.eligible_count,
        "realtime_total_count": realtime_quality.total_count,
        "collect_seconds": config.collect_seconds,
        "final_picks": config.final_picks,
        "pre_candidates": config.pre_candidates,
        "max_symbols_scan": config.max_symbols_scan,
        "long_only": _bool_to_int(config.long_only),
        "min_change_pct": config.min_change_pct,
        "min_gap_pct": config.min_gap_pct,
        "min_prev_turnover": config.min_prev_turnover,
        "min_strength": config.min_strength,
        "min_vol_ratio": config.min_vol_ratio,
        "min_bid_ask_ratio": config.min_bid_ask_ratio,
        "min_pass_conditions": config.min_pass_conditions,
        "min_maintain_ratio": config.min_maintain_ratio,
        "min_realtime_coverage_ratio": config.min_realtime_coverage_ratio,
        "load_universe_sec": timings_sec.get("load_universe_sec", 0.0),
        "stage1_sec": timings_sec.get("stage1_sec", 0.0),
        "fallback_sec": timings_sec.get("fallback_sec", 0.0),
        "realtime_sec": timings_sec.get("realtime_sec", 0.0),
        "refresh_sec": timings_sec.get("refresh_sec", 0.0),
        "scoring_sec": timings_sec.get("scoring_sec", 0.0),
        "total_sec": timings_sec.get("total_sec", 0.0),
        "final_codes": "|".join(item.code for item in final),
    }
    all_run_summary_path = all_dir / "run_summary.csv"
    daily_run_summary_path = daily_dir / "run_summary.csv"
    _append_csv_rows(all_run_summary_path, RUN_SUMMARY_FIELDS, [summary_row])
    _append_csv_rows(daily_run_summary_path, RUN_SUMMARY_FIELDS, [summary_row])

    stage1_scan_rows_raw = getattr(selector, "last_stage1_scan", []) or []
    stage1_scan_rows: List[Dict[str, Any]] = []
    for row in stage1_scan_rows_raw:
        stage1_scan_rows.append(
            {
                "run_id": run_id,
                "market": config.market,
                "decision_at": decision_at.isoformat(),
                "scan_index": row.get("scan_index", ""),
                "code": row.get("code", ""),
                "name": row.get("name", ""),
                "current_price": row.get("current_price", ""),
                "open_price": row.get("open_price", ""),
                "change_pct": row.get("change_pct", ""),
                "gap_pct": row.get("gap_pct", ""),
                "prev_close": row.get("prev_close", ""),
                "prev_day_volume": row.get("prev_day_volume", ""),
                "prev_day_turnover": row.get("prev_day_turnover", ""),
                "pass_change": _bool_to_int(row.get("pass_change")),
                "pass_gap": _bool_to_int(row.get("pass_gap")),
                "pass_prev_turnover": _bool_to_int(row.get("pass_prev_turnover")),
                "passed_stage1": _bool_to_int(row.get("passed_stage1")),
                "skip_reason": row.get("skip_reason", ""),
                "min_change_pct": row.get("min_change_pct", ""),
                "min_gap_pct": row.get("min_gap_pct", ""),
                "min_prev_turnover": row.get("min_prev_turnover", ""),
                "long_only": _bool_to_int(row.get("long_only")),
            }
        )
    all_stage1_scan_path = all_dir / "stage1_scan.csv"
    daily_stage1_scan_path = daily_dir / "stage1_scan.csv"
    _append_csv_rows(all_stage1_scan_path, STAGE1_SCAN_FIELDS, stage1_scan_rows)
    _append_csv_rows(daily_stage1_scan_path, STAGE1_SCAN_FIELDS, stage1_scan_rows)

    ranked_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ranked, start=1):
        cond = item.conditions
        metrics = item.metrics
        rt = realtime_stats.get(item.code)
        ranked_rows.append(
            {
                "run_id": run_id,
                "market": config.market,
                "decision_at": decision_at.isoformat(),
                "rank": idx,
                "is_final_pick": _bool_to_int(item.code in final_slot),
                "selected_slot": final_slot.get(item.code, ""),
                "code": item.code,
                "name": item.name,
                "recommendation_score": round(_recommendation_score(item), 6),
                "score": item.score,
                "max_score": item.max_score,
                "passed": _bool_to_int(item.passed),
                "cond_change_pct": _bool_to_int(cond.get("change_pct")),
                "cond_gap_pct": _bool_to_int(cond.get("gap_pct")),
                "cond_prev_turnover": _bool_to_int(cond.get("prev_turnover")),
                "cond_strength_maintained": _bool_to_int(cond.get("strength_maintained")),
                "cond_volume_ratio": _bool_to_int(cond.get("volume_ratio")),
                "cond_bid_ask_maintained": _bool_to_int(cond.get("bid_ask_maintained")),
                "cond_price_above_vwap": _bool_to_int(cond.get("price_above_vwap")),
                "cond_low_not_broken": _bool_to_int(cond.get("low_not_broken")),
                "metric_current_change_pct": metrics.get("current_change_pct"),
                "metric_gap_pct": metrics.get("gap_pct"),
                "metric_prev_day_turnover": metrics.get("prev_day_turnover"),
                "metric_prev_day_volume": metrics.get("prev_day_volume"),
                "metric_strength_avg": metrics.get("strength_avg"),
                "metric_strength_hit_ratio": metrics.get("strength_hit_ratio"),
                "metric_bid_ask_avg": metrics.get("bid_ask_avg"),
                "metric_bid_ask_hit_ratio": metrics.get("bid_ask_hit_ratio"),
                "metric_volume_ratio": metrics.get("volume_ratio"),
                "metric_vwap": metrics.get("vwap"),
                "metric_latest_price": metrics.get("latest_price"),
                "metric_execution_ticks": metrics.get("execution_ticks"),
                "metric_orderbook_ticks": metrics.get("orderbook_ticks"),
                "metric_realtime_eligible": metrics.get("realtime_eligible"),
                "rt_cum_trade_volume": rt.cum_trade_volume if rt is not None else "",
                "rt_cum_trade_value": rt.cum_trade_value if rt is not None else "",
                "rt_latest_acml_volume": rt.latest_acml_volume if rt is not None else "",
                "rt_first_reported_low": rt.first_reported_low if rt is not None else "",
                "rt_low_broken_after_start": _bool_to_int(
                    rt.low_broken_after_start if rt is not None else None
                ),
                "rt_strength_samples": len(rt.strength_values) if rt is not None else "",
                "rt_bid_ask_samples": len(rt.bid_ask_ratios) if rt is not None else "",
            }
        )
    all_ranked_symbols_path = all_dir / "ranked_symbols.csv"
    daily_ranked_symbols_path = daily_dir / "ranked_symbols.csv"
    _append_csv_rows(all_ranked_symbols_path, RANKED_SYMBOL_FIELDS, ranked_rows)
    _append_csv_rows(daily_ranked_symbols_path, RANKED_SYMBOL_FIELDS, ranked_rows)

    run_bundle = {
        "run_id": run_id,
        "run_started_at": run_started_at.isoformat(),
        "decision_at": decision_at.isoformat(),
        "run_finished_at": run_finished_at.isoformat(),
        "market": config.market,
        "exchange": exchange,
        "signal_valid": signal_valid,
        "invalid_reason": invalid_reason,
        "counts": {
            "universe_size": universe_size,
            "stage1_initial_count": len(stage1_initial),
            "fallback_added_count": fallback_added_count,
            "stage1_final_count": len(stage1_final),
            "decision_candidate_count": len(decision_candidates),
            "ranked_count": len(ranked),
            "final_count": len(final),
        },
        "timings_sec": timings_sec,
        "realtime_quality": asdict(realtime_quality),
        "config": _sanitize_config(config),
        "stage1_scan": stage1_scan_rows,
        "stage1_initial_candidates": [asdict(item) for item in stage1_initial],
        "stage1_final_candidates": [asdict(item) for item in stage1_final],
        "decision_candidates": [asdict(item) for item in decision_candidates],
        "realtime_stats_by_code": {
            code: _serialize_realtime_stats(stats) for code, stats in realtime_stats.items()
        },
        "ranked": _ranked_payload(ranked),
        "final": _ranked_payload(final),
    }
    run_bundle_path = runs_dir / f"{run_id}.json"
    run_bundle_path.write_text(
        json.dumps(run_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "run_summary": daily_run_summary_path,
        "stage1_scan": daily_stage1_scan_path,
        "ranked_symbols": daily_ranked_symbols_path,
        "run_bundle": run_bundle_path,
        "all_run_summary": all_run_summary_path,
        "all_stage1_scan": all_stage1_scan_path,
        "all_ranked_symbols": all_ranked_symbols_path,
    }
