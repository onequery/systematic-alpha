from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import List
from zoneinfo import ZoneInfo

from systematic_alpha.helpers import fmt
from systematic_alpha.models import FinalSelection, StrategyConfig
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.selector import DayTradingSelector


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
    realtime_ready: bool,
    final: List[FinalSelection],
    ranked: List[FinalSelection],
) -> None:
    if not config.output_json_path:
        return
    out_path = Path(config.output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "realtime_ready": realtime_ready,
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


def run(config: StrategyConfig) -> None:
    total_started = perf_counter()
    log(
        "Run config: "
        f"collect={config.collect_seconds}s, "
        f"final_picks={config.final_picks}, pre_candidates={config.pre_candidates}, "
        f"max_symbols_scan={config.max_symbols_scan}, "
        f"stage1_log_interval={config.stage1_log_interval}, "
        f"realtime_log_interval={config.realtime_log_interval}s"
    )
    mojito_module = import_mojito_module()
    selector = DayTradingSelector(mojito_module, config)

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

    if not stage1:
        log("No candidates passed stage1 thresholds.")
        return

    stage_started = perf_counter()
    log("[3/4] Collecting realtime data (strength/VWAP/orderbook)...")
    target_codes = [item.code for item in stage1]
    realtime_stats, realtime_ready = selector.collect_realtime(target_codes)
    if realtime_ready:
        log(f"Realtime data received (elapsed {perf_counter() - stage_started:.1f}s). Running full 8-condition scoring.")
    else:
        log(
            f"Realtime execution data not received (elapsed {perf_counter() - stage_started:.1f}s). "
            "Falling back to stage1-only scoring."
        )

    stage_started = perf_counter()
    log("[4/4] Final scoring...")
    ranked = sort_by_recommendation(selector.evaluate(stage1, realtime_stats, realtime_ready))
    log(f"Scoring complete (elapsed {perf_counter() - stage_started:.1f}s). Ranked={len(ranked)}")

    passed = [item for item in ranked if item.passed]
    final = passed[: config.final_picks]
    if len(final) < config.final_picks:
        needed = config.final_picks - len(final)
        final.extend([item for item in ranked if item not in final][:needed])

    print_final_table(final)
    save_json_output(config, realtime_ready, final, ranked)
    print("\nTop codes:", ", ".join(item.code for item in final))
    log(f"Run finished (total elapsed {perf_counter() - total_started:.1f}s)")
