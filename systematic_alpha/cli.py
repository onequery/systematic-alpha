from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from systematic_alpha.helpers import fmt
from systematic_alpha.models import StrategyConfig
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.selector import DayTradingSelector


def save_json_output(config: StrategyConfig, realtime_ready: bool, final, ranked) -> None:
    if not config.output_json_path:
        return
    out_path = Path(config.output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "realtime_ready": realtime_ready,
                "final": [asdict(item) for item in final],
                "all_ranked": [asdict(item) for item in ranked],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")


def print_final_table(final) -> None:
    print("")
    print("Final Picks")
    print("rank code    score  change%  gap%    volRatio strength bid/ask  vwapOK")
    for idx, item in enumerate(final, start=1):
        vol_ratio = item.metrics.get("volume_ratio")
        strength = item.metrics.get("strength_avg")
        bid_ask = item.metrics.get("bid_ask_avg")
        vwap = item.metrics.get("vwap")
        latest = item.metrics.get("latest_price")
        vwap_ok = latest is not None and vwap is not None and latest >= vwap
        print(
            f"{idx:>4} {item.code:<6} {item.score:>2}/{item.max_score:<3} "
            f"{fmt(item.metrics.get('current_change_pct')):>7} "
            f"{fmt(item.metrics.get('gap_pct')):>7} "
            f"{fmt(vol_ratio, 3):>8} "
            f"{fmt(strength):>8} "
            f"{fmt(bid_ask, 3):>7} "
            f"{'Y' if vwap_ok else 'N':>6}"
        )


def run(config: StrategyConfig) -> None:
    mojito_module = import_mojito_module()
    selector = DayTradingSelector(mojito_module, config)

    print("[1/4] Loading universe...", flush=True)
    codes, names = selector.load_universe()
    if not codes:
        raise RuntimeError("No symbols loaded. Provide --universe-file or check fetch_symbols().")
    print(f"Universe size: {len(codes)}", flush=True)

    print("[2/4] Stage1 filtering (change/gap/prev turnover)...", flush=True)
    stage1 = selector.build_stage1_candidates(codes, names)
    print(f"Stage1 candidates: {len(stage1)}", flush=True)
    if not stage1:
        print("No candidates passed stage1 thresholds.")
        return

    print("[3/4] Collecting realtime data (strength/VWAP/orderbook)...", flush=True)
    target_codes = [item.code for item in stage1]
    realtime_stats, realtime_ready = selector.collect_realtime(target_codes)
    if realtime_ready:
        print("Realtime data received. Running full 8-condition scoring.", flush=True)
    else:
        print("Realtime execution data not received. Falling back to stage1-only scoring.", flush=True)

    print("[4/4] Final scoring...", flush=True)
    ranked = selector.evaluate(stage1, realtime_stats, realtime_ready)

    passed = [item for item in ranked if item.passed]
    final = passed[: config.final_picks]
    if len(final) < config.final_picks:
        needed = config.final_picks - len(final)
        final.extend([item for item in ranked if item not in final][:needed])

    print_final_table(final)
    save_json_output(config, realtime_ready, final, ranked)
    print("\nTop codes:", ", ".join(item.code for item in final))
