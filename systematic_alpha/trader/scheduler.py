from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from systematic_alpha.trader.config import TraderConfig


KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


def parse_hhmmss(raw: str) -> time:
    text = str(raw or "").strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except Exception:
            continue
    raise ValueError(f"invalid time format: {raw}")


def combine_kst(day: date, raw: str) -> datetime:
    return datetime.combine(day, parse_hhmmss(raw), tzinfo=KST)


def combine_et(day: date, raw: str) -> datetime:
    return datetime.combine(day, parse_hhmmss(raw), tzinfo=ET)


def now_kst() -> datetime:
    return datetime.now(KST)


def now_et() -> datetime:
    return datetime.now(ET)


def trade_date_kst() -> str:
    return now_kst().strftime("%Y%m%d")


def trade_date_et() -> str:
    return now_et().strftime("%Y%m%d")


def is_weekday_kst(ref: datetime | None = None) -> bool:
    dt = ref.astimezone(KST) if ref else now_kst()
    return dt.weekday() < 5


def is_weekday_et(ref: datetime | None = None) -> bool:
    dt = ref.astimezone(ET) if ref else now_et()
    return dt.weekday() < 5


@dataclass(frozen=True)
class MarketWindow:
    market: str
    open_time: datetime
    close_time: datetime


def kr_window(ref: datetime | None = None) -> MarketWindow:
    dt = ref.astimezone(KST) if ref else now_kst()
    day = dt.date()
    open_dt = datetime.combine(day, time(9, 0, 0), tzinfo=KST)
    close_dt = datetime.combine(day, time(15, 30, 0), tzinfo=KST)
    return MarketWindow(market="KR", open_time=open_dt, close_time=close_dt)


def us_window(ref: datetime | None = None) -> MarketWindow:
    dt = ref.astimezone(ET) if ref else now_et()
    day = dt.date()
    open_dt = datetime.combine(day, time(9, 30, 0), tzinfo=ET)
    close_dt = datetime.combine(day, time(16, 0, 0), tzinfo=ET)
    return MarketWindow(market="US", open_time=open_dt, close_time=close_dt)


def is_market_open(market: str, ref: datetime | None = None) -> bool:
    mk = str(market or "").strip().upper()
    if mk == "KR":
        dt = ref.astimezone(KST) if ref else now_kst()
        w = kr_window(dt)
        return w.open_time <= dt < w.close_time
    if mk == "US":
        dt = ref.astimezone(ET) if ref else now_et()
        w = us_window(dt)
        return w.open_time <= dt < w.close_time
    return False


@dataclass(frozen=True)
class LiquidationPlan:
    market: str
    primary_start: datetime
    primary_end: datetime
    retry_time: datetime
    final_check: datetime


def kr_liquidation_plan(cfg: TraderConfig, ref: datetime | None = None) -> LiquidationPlan:
    dt = ref.astimezone(KST) if ref else now_kst()
    day = dt.date()
    start = combine_kst(day, cfg.kr_liquidation_start_kst)
    end = start + timedelta(minutes=5)
    retry = combine_kst(day, cfg.kr_liquidation_retry_kst)
    final_check = combine_kst(day, cfg.kr_liquidation_final_kst)
    return LiquidationPlan("KR", start, end, retry, final_check)


def us_liquidation_plan(cfg: TraderConfig, ref: datetime | None = None) -> LiquidationPlan:
    dt = ref.astimezone(ET) if ref else now_et()
    w = us_window(dt)
    start = w.close_time - timedelta(minutes=cfg.us_liquidation_lead_minutes)
    end = start + timedelta(minutes=5)
    retry = w.close_time - timedelta(minutes=cfg.us_liquidation_retry_minutes)
    final_check = w.close_time - timedelta(seconds=cfg.us_liquidation_final_seconds)
    return LiquidationPlan("US", start, end, retry, final_check)


def in_window(start: datetime, end: datetime, ref: datetime) -> bool:
    return start <= ref < end


def precompute_window(cfg: TraderConfig, ref: datetime | None = None) -> tuple[datetime, datetime]:
    dt = ref.astimezone(KST) if ref else now_kst()
    day = dt.date()
    return combine_kst(day, cfg.precompute_start_kst), combine_kst(day, cfg.precompute_end_kst)


def budget_snapshot_time(cfg: TraderConfig, ref: datetime | None = None) -> datetime:
    dt = ref.astimezone(KST) if ref else now_kst()
    return combine_kst(dt.date(), cfg.budget_snapshot_kst)


def should_run_precompute(cfg: TraderConfig, ref: datetime | None = None) -> bool:
    dt = ref.astimezone(KST) if ref else now_kst()
    if not is_weekday_kst(dt):
        return False
    start, end = precompute_window(cfg, dt)
    return start <= dt <= end


def should_run_budget_snapshot(cfg: TraderConfig, ref: datetime | None = None, grace_minutes: int = 20) -> bool:
    dt = ref.astimezone(KST) if ref else now_kst()
    if not is_weekday_kst(dt):
        return False
    snap = budget_snapshot_time(cfg, dt)
    return snap <= dt <= snap + timedelta(minutes=max(1, grace_minutes))


def liquidation_phase(cfg: TraderConfig, market: str, ref: datetime | None = None) -> str | None:
    mk = str(market or "").upper()
    if mk == "KR":
        dt = ref.astimezone(KST) if ref else now_kst()
        if not is_weekday_kst(dt):
            return None
        plan = kr_liquidation_plan(cfg, dt)
        if in_window(plan.primary_start, plan.primary_end, dt):
            return "primary"
        if dt >= plan.retry_time and dt < plan.final_check:
            return "retry"
        if dt >= plan.final_check and dt < (plan.final_check + timedelta(minutes=5)):
            return "final_check"
        return None
    if mk == "US":
        dt = ref.astimezone(ET) if ref else now_et()
        if not is_weekday_et(dt):
            return None
        plan = us_liquidation_plan(cfg, dt)
        if in_window(plan.primary_start, plan.primary_end, dt):
            return "primary"
        if dt >= plan.retry_time and dt < plan.final_check:
            return "retry"
        if dt >= plan.final_check and dt < (plan.final_check + timedelta(minutes=5)):
            return "final_check"
    return None
