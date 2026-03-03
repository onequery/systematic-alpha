from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Tuple

import pytest

from systematic_alpha.dotenv import load_env_stack
from systematic_alpha.mojito_loader import import_mojito_module
from systematic_alpha.selector_kr import DayTradingSelector
from systematic_alpha.selector_us import USDayTradingSelector
from systematic_alpha.trader.config import load_trader_config
from systematic_alpha.trader.precompute import _fetch_ohlcv_rows_with_diag
from systematic_alpha.trader.selector_bridge import build_strategy_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _retry(callable_fn: Callable[[], Any], attempts: int = 3, delay_sec: float = 1.5) -> Any:
    last_error: Exception | None = None
    for idx in range(max(1, int(attempts))):
        try:
            return callable_fn()
        except Exception as exc:  # pragma: no cover - exercised only in live test environments
            last_error = exc
            if idx + 1 >= attempts:
                break
            time.sleep(max(0.0, float(delay_sec)))
    if last_error is None:
        raise RuntimeError("retry_failed_without_exception")
    raise last_error


def _build_selectors() -> Tuple[DayTradingSelector, USDayTradingSelector]:
    root = _repo_root()
    load_env_stack(root, files=("config/trader.config", ".env"))
    cfg = load_trader_config(root)
    mojito = import_mojito_module()

    kr_cfg = build_strategy_config(cfg=cfg, market="KR")
    us_cfg = build_strategy_config(cfg=cfg, market="US")
    return DayTradingSelector(mojito, kr_cfg), USDayTradingSelector(mojito, us_cfg)


@pytest.mark.live_api
def test_us_objective_remote_source_is_available() -> None:
    _, us_selector = _build_selectors()
    symbols, names = _retry(lambda: us_selector._fetch_sp500_remote(), attempts=2, delay_sec=1.0)
    assert isinstance(symbols, list) and len(symbols) > 100
    assert isinstance(names, dict)


@pytest.mark.live_api
def test_kr_symbol_master_source_is_available() -> None:
    kr_selector, _ = _build_selectors()

    def _fetch_count() -> int:
        df = kr_selector.broker.fetch_symbols()
        if hasattr(df, "shape") and len(df.shape) >= 1:
            return int(df.shape[0])
        if hasattr(df, "__len__"):
            return int(len(df))  # type: ignore[arg-type]
        return 0

    count = _retry(_fetch_count, attempts=2, delay_sec=1.0)
    assert count > 1000


@pytest.mark.live_api
def test_us_spy_has_enough_bars_for_ma20() -> None:
    _, us_selector = _build_selectors()

    def _load_rows_with_diag() -> tuple[list[dict], dict]:
        rows, diag = _fetch_ohlcv_rows_with_diag(us_selector, "US", "SPY")
        return rows, diag

    rows, diag = _retry(_load_rows_with_diag, attempts=3, delay_sec=1.5)
    assert isinstance(rows, list), f"SPY rows is not a list: diag={diag!r}"
    assert len(rows) >= 20, f"SPY OHLCV bars insufficient: got={len(rows)} required=20 diag={diag!r}"
    closes = [r.get("close") for r in rows if r.get("close") is not None]
    assert len(closes) >= 20, f"SPY close bars insufficient: got={len(closes)} required=20 diag={diag!r}"
