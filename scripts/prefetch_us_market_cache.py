from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systematic_alpha.credentials import load_credentials
from systematic_alpha.dotenv import load_env_stack
from systematic_alpha.models import StrategyConfig
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.selector_us import USDayTradingSelector


def build_prefetch_config(
    *,
    api_key: str,
    api_secret: str,
    acc_no: str,
    user_id: str | None,
    us_exchange: str,
    us_universe_size: int,
    max_symbols_scan: int,
) -> StrategyConfig:
    return StrategyConfig(
        market="US",
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no,
        user_id=user_id,
        mock=False,
        us_exchange=us_exchange,
        us_poll_interval=2.0,
        kr_universe_size=500,
        us_universe_size=max(50, us_universe_size),
        universe_file=None,
        max_symbols_scan=max(50, max_symbols_scan),
        pre_candidates=40,
        final_picks=3,
        collect_seconds=0,
        rest_sleep_sec=0.03,
        min_change_pct=3.0,
        min_gap_pct=2.0,
        min_prev_turnover=10_000_000_000,
        min_strength=100.0,
        min_vol_ratio=0.10,
        min_bid_ask_ratio=1.2,
        min_pass_conditions=5,
        min_maintain_ratio=0.6,
        min_strength_samples=3,
        min_bid_ask_samples=3,
        long_only=True,
        min_exec_ticks=30,
        min_orderbook_ticks=30,
        min_realtime_cum_volume=1.0,
        min_realtime_coverage_ratio=0.8,
        invalidate_on_low_coverage=True,
        stage1_log_interval=20,
        realtime_log_interval=10,
        overnight_report_path=None,
        output_json_path=None,
        analytics_dir=None,
        enable_analytics_log=False,
        test_assume_open=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefetch US market caches (universe load + prev-day stats cache)."
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--key-file", type=str, default=None)
    parser.add_argument("--us-exchange", type=str, default="NASD")
    parser.add_argument("--us-universe-size", type=int, default=500)
    parser.add_argument("--max-symbols-scan", type=int, default=500)
    parser.add_argument("--min-success-ratio", type=float, default=0.2)
    parser.add_argument("--min-success-count", type=int, default=20)
    parser.add_argument("--force-refresh", action="store_true", default=False)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"[prefetch-us-cache] project root not found: {project_root}", flush=True)
        return 1

    os.chdir(project_root)
    load_env_stack(project_root)

    api_key, api_secret, acc_no, user_id = load_credentials(args.key_file)
    config = build_prefetch_config(
        api_key=api_key,
        api_secret=api_secret,
        acc_no=acc_no,
        user_id=user_id,
        us_exchange=args.us_exchange,
        us_universe_size=args.us_universe_size,
        max_symbols_scan=args.max_symbols_scan,
    )
    mojito_module = import_mojito_module()
    selector = USDayTradingSelector(mojito_module, config)

    prev_cache = selector._prev_stats_cache_path()
    if args.force_refresh and prev_cache.exists():
        prev_cache.unlink(missing_ok=True)
        print(f"[prefetch-us-cache] removed existing prev cache: {prev_cache}", flush=True)

    print(
        f"[prefetch-us-cache] start: exchange={config.us_exchange}, "
        f"us_universe_size={config.us_universe_size}, max_symbols_scan={config.max_symbols_scan}",
        flush=True,
    )
    codes, _ = selector.load_universe()
    if not codes:
        print("[prefetch-us-cache] failed: no symbols loaded", flush=True)
        return 2

    prev_success, prev_total = selector.prefetch_prev_day_stats(codes, force_refresh=args.force_refresh)
    success_ratio = (float(prev_success) / float(prev_total)) if prev_total > 0 else 0.0
    print(
        (
            "[prefetch-us-cache] prev-day cache: "
            f"success={prev_success}/{prev_total} ({success_ratio:.3f}), cache={prev_cache}"
        ),
        flush=True,
    )

    if prev_total <= 0:
        print("[prefetch-us-cache] failed: prev-day cache scan had zero symbols", flush=True)
        return 3

    min_success_count = max(1, int(args.min_success_count))
    min_success_ratio = max(0.0, min(1.0, float(args.min_success_ratio)))
    if prev_success < min_success_count or success_ratio < min_success_ratio:
        print(
            (
                "[prefetch-us-cache] failed: prev-day cache coverage too low "
                f"(success={prev_success}, total={prev_total}, ratio={success_ratio:.3f}, "
                f"min_success_count={min_success_count}, min_success_ratio={min_success_ratio:.3f})"
            ),
            flush=True,
        )
        if hasattr(selector, "get_api_diagnostics"):
            try:
                diag = selector.get_api_diagnostics()  # type: ignore[attr-defined]
                print(f"[prefetch-us-cache] api_diagnostics={diag}", flush=True)
            except Exception:
                pass
        return 4

    if not prev_cache.exists():
        print(f"[prefetch-us-cache] failed: prev cache not created ({prev_cache})", flush=True)
        return 5

    print(
        f"[prefetch-us-cache] success: selected={len(codes)}, cache={prev_cache}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
