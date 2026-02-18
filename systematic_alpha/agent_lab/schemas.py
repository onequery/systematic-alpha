from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

STATUS_SIGNAL_OK = "SIGNAL_OK"
STATUS_MARKET_CLOSED = "MARKET_CLOSED"
STATUS_INVALID_SIGNAL = "INVALID_SIGNAL"
STATUS_DATA_QUALITY_LOW = "DATA_QUALITY_LOW"

PROPOSAL_STATUS_PENDING_APPROVAL = "PENDING_APPROVAL"
PROPOSAL_STATUS_APPROVED = "APPROVED"
PROPOSAL_STATUS_REJECTED = "REJECTED"
PROPOSAL_STATUS_BLOCKED = "BLOCKED"
PROPOSAL_STATUS_EXECUTED = "EXECUTED"

ORDER_SIDE_BUY = "BUY"
ORDER_SIDE_SELL = "SELL"

ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_LIMIT = "LIMIT"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class AgentProfile:
    agent_id: str
    name: str
    role: str
    philosophy: str
    allocated_capital_krw: float
    risk_style: str
    constraints: Dict[str, Any]


@dataclass
class StrategyVersion:
    strategy_version_id: Optional[int]
    agent_id: str
    version_tag: str
    params: Dict[str, Any]
    promoted: bool
    notes: str = ""
    created_at: str = field(default_factory=now_iso)


@dataclass
class SessionSignal:
    market: str
    session_date: str
    generated_at: str
    signal_valid: bool
    status_code: str
    invalid_reason: str
    source_json_path: str
    payload: Dict[str, Any]


@dataclass
class ProposedOrder:
    market: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    limit_price: Optional[float]
    reference_price: float
    signal_rank: int
    recommendation_score: float
    rationale: str


@dataclass
class OrderProposal:
    proposal_uuid: str
    agent_id: str
    market: str
    session_date: str
    strategy_version_id: int
    status: str
    blocked_reason: str
    orders: List[ProposedOrder]
    rationale: str
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)


@dataclass
class RiskDecision:
    allowed: bool
    blocked_reason: str
    accepted_orders: List[ProposedOrder]
    dropped_orders: List[Dict[str, Any]]
    metrics: Dict[str, Any]


@dataclass
class AgentReview:
    agent_id: str
    date: str
    equity_krw: float
    return_pct: float
    mdd: float
    volatility: float
    win_rate: float
    profit_factor: float
    turnover: float
    max_consecutive_loss: int
    notes: str


@dataclass
class WeeklyDecision:
    week_id: str
    champion_agent_id: str
    score_board: Dict[str, float]
    reasoning: str
    promoted_versions: Dict[str, str]
