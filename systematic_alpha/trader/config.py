from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from systematic_alpha.dotenv import load_env_stack


TRUE_SET = {"1", "true", "yes", "y", "on"}


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in TRUE_SET


def _int_env(name: str, default: int, minimum: int | None = None) -> int:
    try:
        out = int(float(os.getenv(name, str(default)) or default))
    except Exception:
        out = int(default)
    if minimum is not None:
        out = max(minimum, out)
    return out


def _float_env(name: str, default: float, minimum: float | None = None) -> float:
    try:
        out = float(os.getenv(name, str(default)) or default)
    except Exception:
        out = float(default)
    if minimum is not None:
        out = max(minimum, out)
    return out


def _csv_env(name: str, default: Iterable[str]) -> List[str]:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return [str(x).strip() for x in default if str(x).strip()]
    out: List[str] = []
    seen = set()
    for tok in raw.split(","):
        item = str(tok).strip()
        if not item:
            continue
        up = item.upper()
        if up in seen:
            continue
        seen.add(up)
        out.append(up)
    return out


def _text_env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or default).strip()


def normalize_profile(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "prod"
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = text.strip("_")
    if not text:
        return "prod"
    return text


def profile_suffix(profile: str) -> str:
    name = normalize_profile(profile)
    if name in {"prod", "main", "default"}:
        return ""
    return f"_{name}"


def profile_dir(root: Path, bucket: str, profile: str) -> Path:
    suffix = profile_suffix(profile)
    return Path(root) / str(bucket) / f"trader{suffix}"


def state_dir_from_cfg(cfg: object) -> Path:
    val = getattr(cfg, "state_dir", None)
    if val:
        return Path(val)
    root = Path(getattr(cfg, "project_root", "."))
    profile = normalize_profile(str(getattr(cfg, "profile", "prod")))
    return profile_dir(root, "state", profile)


def out_dir_from_cfg(cfg: object) -> Path:
    val = getattr(cfg, "out_dir", None)
    if val:
        return Path(val)
    root = Path(getattr(cfg, "project_root", "."))
    profile = normalize_profile(str(getattr(cfg, "profile", "prod")))
    return profile_dir(root, "out", profile)


def logs_dir_from_cfg(cfg: object) -> Path:
    val = getattr(cfg, "logs_dir", None)
    if val:
        return Path(val)
    root = Path(getattr(cfg, "project_root", "."))
    profile = normalize_profile(str(getattr(cfg, "profile", "prod")))
    return profile_dir(root, "logs", profile)


@dataclass(frozen=True)
class TraderConfig:
    project_root: Path
    profile: str
    state_dir: Path
    out_dir: Path
    logs_dir: Path

    execution_mode: str
    enabled_markets: List[str]
    us_sync_exchanges: List[str]
    us_require_all_exchanges: bool

    k: float
    per_trade_ratio: float
    max_positions_kr: int
    max_positions_us: int
    max_positions_total: int
    candidates_max_kr: int
    candidates_max_us: int

    precompute_start_kst: str
    precompute_end_kst: str
    cooldown_start_kst: str
    cooldown_end_kst: str
    budget_snapshot_kst: str

    kr_liquidation_start_kst: str
    kr_liquidation_retry_kst: str
    kr_liquidation_final_kst: str
    us_liquidation_lead_minutes: int
    us_liquidation_retry_minutes: int
    us_liquidation_final_seconds: int

    broker_global_serialize: bool
    broker_global_min_interval_sec: float
    us_dayornight_call_spacing_sec: float
    us_exchange_spacing_sec: float
    rate_limit_retries: int
    rate_limit_backoff_sec: float
    rate_limit_backoff_max_sec: float

    strict_sync: bool
    sync_max_staleness_sec: int

    rest_sleep_sec: float

    market_filter_symbol_kr: str
    market_filter_symbol_us: str
    market_filter_ma_days: int

    poll_seconds: int
    strategy_cycle_seconds: int
    fallback_poll_seconds: int
    cooldown_minutes: int

    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_thread_id: str
    telegram_disable_notification: bool

    @property
    def use_mock(self) -> bool:
        return "mock" in self.execution_mode

    def market_enabled(self, market: str) -> bool:
        return str(market or "").strip().upper() in set(self.enabled_markets)


def load_trader_config(project_root: str | Path = ".") -> TraderConfig:
    root = Path(project_root).resolve()
    load_env_stack(root, files=("config/trader.config", ".env"))
    profile = normalize_profile(_text_env("TRADER_PROFILE", "prod"))
    state_dir = profile_dir(root, "state", profile)
    out_dir = profile_dir(root, "out", profile)
    logs_dir = profile_dir(root, "logs", profile)

    token = _text_env("TELEGRAM_BOT_TOKEN", "")
    if token.lower().startswith("bot"):
        token = token[3:]
    chat_id = _text_env("TELEGRAM_CHAT_ID", "")

    enabled_markets = _csv_env("TRADER_ORDER_ENABLED_MARKETS", ["KR", "US"])
    enabled_markets = [m for m in enabled_markets if m in {"KR", "US"}]
    if not enabled_markets:
        enabled_markets = ["KR", "US"]

    us_sync_exchanges = _csv_env("TRADER_US_SYNC_EXCHANGES", ["NASD", "NYSE", "AMEX"])
    us_sync_exchanges = [m for m in us_sync_exchanges if m in {"NASD", "NYSE", "AMEX"}]
    if not us_sync_exchanges:
        us_sync_exchanges = ["NASD", "NYSE", "AMEX"]

    return TraderConfig(
        project_root=root,
        profile=profile,
        state_dir=state_dir,
        out_dir=out_dir,
        logs_dir=logs_dir,
        execution_mode=_text_env("TRADER_EXECUTION_MODE", "mojito_mock").lower(),
        enabled_markets=enabled_markets,
        us_sync_exchanges=us_sync_exchanges,
        us_require_all_exchanges=_bool_env("TRADER_US_REQUIRE_ALL_EXCHANGES", True),
        k=_float_env("TRADER_BREAKOUT_K", 0.5, 0.0),
        per_trade_ratio=_float_env("TRADER_PER_TRADE_RATIO", 0.12, 0.001),
        max_positions_kr=_int_env("TRADER_MAX_POS_KR", 3, 1),
        max_positions_us=_int_env("TRADER_MAX_POS_US", 3, 1),
        max_positions_total=_int_env("TRADER_MAX_POS_TOTAL", 6, 1),
        candidates_max_kr=_int_env("TRADER_CANDIDATES_MAX_KR", 20, 1),
        candidates_max_us=_int_env("TRADER_CANDIDATES_MAX_US", 20, 1),
        precompute_start_kst=_text_env("TRADER_PRECOMPUTE_START_KST", "08:05"),
        precompute_end_kst=_text_env("TRADER_PRECOMPUTE_END_KST", "08:33"),
        cooldown_start_kst=_text_env("TRADER_COOLDOWN_START_KST", "08:33"),
        cooldown_end_kst=_text_env("TRADER_COOLDOWN_END_KST", "08:38"),
        budget_snapshot_kst=_text_env("TRADER_BUDGET_SNAPSHOT_KST", "08:40"),
        kr_liquidation_start_kst=_text_env("TRADER_KR_LIQUIDATION_START_KST", "15:10"),
        kr_liquidation_retry_kst=_text_env("TRADER_KR_LIQUIDATION_RETRY_KST", "15:18"),
        kr_liquidation_final_kst=_text_env("TRADER_KR_LIQUIDATION_FINAL_KST", "15:19:30"),
        us_liquidation_lead_minutes=_int_env("TRADER_US_LIQUIDATION_LEAD_MINUTES", 20, 1),
        us_liquidation_retry_minutes=_int_env("TRADER_US_LIQUIDATION_RETRY_MINUTES", 5, 1),
        us_liquidation_final_seconds=_int_env("TRADER_US_LIQUIDATION_FINAL_SECONDS", 30, 1),
        broker_global_serialize=_bool_env("TRADER_BROKER_GLOBAL_SERIALIZE", True),
        broker_global_min_interval_sec=_float_env("TRADER_BROKER_GLOBAL_MIN_INTERVAL_SEC", 1.2, 0.0),
        us_dayornight_call_spacing_sec=_float_env("TRADER_US_DAYORNIGHT_CALL_SPACING_SEC", 0.4, 0.0),
        us_exchange_spacing_sec=_float_env("TRADER_US_EXCHANGE_SPACING_SEC", 2.0, 0.0),
        rate_limit_retries=_int_env("TRADER_RATE_LIMIT_RETRIES", 3, 0),
        rate_limit_backoff_sec=_float_env("TRADER_RATE_LIMIT_BACKOFF_SEC", 2.0, 0.0),
        rate_limit_backoff_max_sec=_float_env("TRADER_RATE_LIMIT_BACKOFF_MAX_SEC", 20.0, 0.1),
        strict_sync=_bool_env("TRADER_SYNC_STRICT", True),
        sync_max_staleness_sec=_int_env("TRADER_SYNC_MAX_STALENESS_SEC", 30, 1),
        rest_sleep_sec=_float_env("TRADER_REST_SLEEP_SEC", 0.06, 0.0),
        market_filter_symbol_kr=_text_env("TRADER_MARKET_FILTER_SYMBOL_KR", "069500"),
        market_filter_symbol_us=_text_env("TRADER_MARKET_FILTER_SYMBOL_US", "SPY").upper(),
        market_filter_ma_days=_int_env("TRADER_MARKET_FILTER_MA_DAYS", 20, 2),
        poll_seconds=_int_env("TRADER_POLL_SECONDS", 5, 1),
        strategy_cycle_seconds=_int_env("TRADER_STRATEGY_CYCLE_SECONDS", 7200, 60),
        fallback_poll_seconds=_int_env("TRADER_FALLBACK_POLL_SECONDS", 3, 1),
        cooldown_minutes=_int_env("TRADER_ENTRY_COOLDOWN_MINUTES", 10, 1),
        telegram_enabled=_bool_env("TELEGRAM_ENABLED", True) and bool(token and chat_id),
        telegram_bot_token=token,
        telegram_chat_id=chat_id,
        telegram_thread_id=_text_env("TELEGRAM_THREAD_ID", ""),
        telegram_disable_notification=_bool_env("TELEGRAM_DISABLE_NOTIFICATION", False),
    )
