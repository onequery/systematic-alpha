from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class AgentLabStorage:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self):
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def init_schema(self) -> None:
        with self.tx():
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    philosophy TEXT NOT NULL,
                    allocated_capital_krw REAL NOT NULL,
                    risk_style TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS strategy_versions (
                    strategy_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    version_tag TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    promoted INTEGER NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
                );
                CREATE INDEX IF NOT EXISTS idx_strategy_versions_agent
                    ON strategy_versions(agent_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS session_signals (
                    session_signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    signal_valid INTEGER NOT NULL,
                    status_code TEXT NOT NULL,
                    invalid_reason TEXT NOT NULL,
                    source_json_path TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_session_signals_market_date
                    ON session_signals(market, session_date, generated_at DESC);

                CREATE TABLE IF NOT EXISTS order_proposals (
                    proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_uuid TEXT NOT NULL UNIQUE,
                    agent_id TEXT NOT NULL,
                    market TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    strategy_version_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    blocked_reason TEXT NOT NULL DEFAULT '',
                    orders_json TEXT NOT NULL,
                    rationale TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(agent_id) REFERENCES agents(agent_id),
                    FOREIGN KEY(strategy_version_id) REFERENCES strategy_versions(strategy_version_id)
                );
                CREATE INDEX IF NOT EXISTS idx_order_proposals_agent_date
                    ON order_proposals(agent_id, session_date, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_order_proposals_status
                    ON order_proposals(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS order_approvals (
                    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id INTEGER NOT NULL,
                    approved_by TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(proposal_id) REFERENCES order_proposals(proposal_id)
                );

                CREATE TABLE IF NOT EXISTS paper_orders (
                    paper_order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    limit_price REAL,
                    reference_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    broker_order_id TEXT NOT NULL DEFAULT '',
                    broker_response_json TEXT NOT NULL DEFAULT '',
                    submitted_at TEXT NOT NULL,
                    FOREIGN KEY(proposal_id) REFERENCES order_proposals(proposal_id),
                    FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
                );
                CREATE INDEX IF NOT EXISTS idx_paper_orders_agent_ts
                    ON paper_orders(agent_id, submitted_at DESC);

                CREATE TABLE IF NOT EXISTS paper_fills (
                    paper_fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_order_id INTEGER NOT NULL,
                    fill_price REAL NOT NULL,
                    fill_quantity REAL NOT NULL,
                    fill_value_krw REAL NOT NULL,
                    fx_rate REAL NOT NULL,
                    filled_at TEXT NOT NULL,
                    FOREIGN KEY(paper_order_id) REFERENCES paper_orders(paper_order_id)
                );
                CREATE INDEX IF NOT EXISTS idx_paper_fills_ts
                    ON paper_fills(filled_at DESC);

                CREATE TABLE IF NOT EXISTS positions (
                    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    market_value_krw REAL NOT NULL,
                    unrealized_pnl_krw REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(agent_id, market, symbol),
                    FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
                );

                CREATE TABLE IF NOT EXISTS equity_curve (
                    equity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    cash_krw REAL NOT NULL,
                    invested_krw REAL NOT NULL,
                    equity_krw REAL NOT NULL,
                    realized_pnl_krw REAL NOT NULL,
                    unrealized_pnl_krw REAL NOT NULL,
                    drawdown REAL NOT NULL,
                    volatility REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    turnover REAL NOT NULL,
                    max_consecutive_loss INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(agent_id, as_of_date),
                    FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
                );
                CREATE INDEX IF NOT EXISTS idx_equity_curve_agent_date
                    ON equity_curve(agent_id, as_of_date DESC);

                CREATE TABLE IF NOT EXISTS daily_reviews (
                    daily_review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_date TEXT NOT NULL UNIQUE,
                    summary_json TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS weekly_councils (
                    weekly_council_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    week_id TEXT NOT NULL UNIQUE,
                    champion_agent_id TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_memories (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(agent_id) REFERENCES agents(agent_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_ts
                    ON agent_memories(agent_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS state_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_state_events_type_ts
                    ON state_events(event_type, created_at DESC);
                """
            )

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Tuple[Any, ...]]) -> sqlite3.Cursor:
        return self._conn.executemany(sql, rows)

    def query_all(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def query_one(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def log_event(self, event_type: str, payload: Dict[str, Any], created_at: str) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO state_events(event_type, payload_json, created_at)
                VALUES (?, ?, ?)
                """,
                (event_type, json.dumps(payload, ensure_ascii=False), created_at),
            )

    def upsert_agent(
        self,
        agent_id: str,
        name: str,
        role: str,
        philosophy: str,
        allocated_capital_krw: float,
        risk_style: str,
        constraints: Dict[str, Any],
        created_at: str,
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO agents(
                    agent_id, name, role, philosophy, allocated_capital_krw,
                    risk_style, constraints_json, created_at, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(agent_id) DO UPDATE SET
                    name=excluded.name,
                    role=excluded.role,
                    philosophy=excluded.philosophy,
                    allocated_capital_krw=excluded.allocated_capital_krw,
                    risk_style=excluded.risk_style,
                    constraints_json=excluded.constraints_json,
                    is_active=1
                """,
                (
                    agent_id,
                    name,
                    role,
                    philosophy,
                    float(allocated_capital_krw),
                    risk_style,
                    json.dumps(constraints, ensure_ascii=False),
                    created_at,
                ),
            )

    def list_agents(self) -> List[Dict[str, Any]]:
        rows = self.query_all(
            """
            SELECT *
            FROM agents
            WHERE is_active = 1
            ORDER BY agent_id
            """
        )
        for row in rows:
            row["constraints"] = json.loads(row.pop("constraints_json"))
        return rows

    def insert_strategy_version(
        self,
        agent_id: str,
        version_tag: str,
        params: Dict[str, Any],
        promoted: bool,
        notes: str,
        created_at: str,
    ) -> int:
        with self.tx():
            if promoted:
                self.execute(
                    """
                    UPDATE strategy_versions
                    SET promoted = 0
                    WHERE agent_id = ?
                    """,
                    (agent_id,),
                )
            cur = self.execute(
                """
                INSERT INTO strategy_versions(
                    agent_id, version_tag, params_json, promoted, notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    version_tag,
                    json.dumps(params, ensure_ascii=False),
                    1 if promoted else 0,
                    notes,
                    created_at,
                ),
            )
            return int(cur.lastrowid)

    def get_active_strategy(self, agent_id: str) -> Optional[Dict[str, Any]]:
        row = self.query_one(
            """
            SELECT *
            FROM strategy_versions
            WHERE agent_id = ?
            ORDER BY promoted DESC, created_at DESC
            LIMIT 1
            """,
            (agent_id,),
        )
        if row is None:
            return None
        row["params"] = json.loads(row.pop("params_json"))
        return row

    def list_strategy_versions(self, agent_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.query_all(
            """
            SELECT *
            FROM strategy_versions
            WHERE agent_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (agent_id, int(limit)),
        )
        for row in rows:
            row["params"] = json.loads(row.pop("params_json"))
        return rows

    def insert_session_signal(
        self,
        market: str,
        session_date: str,
        generated_at: str,
        signal_valid: bool,
        status_code: str,
        invalid_reason: str,
        source_json_path: str,
        payload: Dict[str, Any],
        created_at: str,
    ) -> int:
        with self.tx():
            cur = self.execute(
                """
                INSERT INTO session_signals(
                    market, session_date, generated_at, signal_valid, status_code,
                    invalid_reason, source_json_path, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market,
                    session_date,
                    generated_at,
                    1 if signal_valid else 0,
                    status_code,
                    invalid_reason,
                    source_json_path,
                    json.dumps(payload, ensure_ascii=False),
                    created_at,
                ),
            )
            return int(cur.lastrowid)

    def get_latest_session_signal(self, market: str, session_date: str) -> Optional[Dict[str, Any]]:
        row = self.query_one(
            """
            SELECT *
            FROM session_signals
            WHERE market = ? AND session_date = ?
            ORDER BY generated_at DESC, session_signal_id DESC
            LIMIT 1
            """,
            (market, session_date),
        )
        if row is None:
            return None
        row["payload"] = json.loads(row.pop("payload_json"))
        row["signal_valid"] = bool(row["signal_valid"])
        return row

    def insert_order_proposal(
        self,
        proposal_uuid: str,
        agent_id: str,
        market: str,
        session_date: str,
        strategy_version_id: int,
        status: str,
        blocked_reason: str,
        orders: List[Dict[str, Any]],
        rationale: str,
        created_at: str,
        updated_at: str,
    ) -> int:
        with self.tx():
            cur = self.execute(
                """
                INSERT INTO order_proposals(
                    proposal_uuid, agent_id, market, session_date, strategy_version_id,
                    status, blocked_reason, orders_json, rationale, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal_uuid,
                    agent_id,
                    market,
                    session_date,
                    int(strategy_version_id),
                    status,
                    blocked_reason,
                    json.dumps(orders, ensure_ascii=False),
                    rationale,
                    created_at,
                    updated_at,
                ),
            )
            return int(cur.lastrowid)

    def get_order_proposal_by_uuid(self, proposal_uuid: str) -> Optional[Dict[str, Any]]:
        row = self.query_one(
            """
            SELECT *
            FROM order_proposals
            WHERE proposal_uuid = ?
            """,
            (proposal_uuid,),
        )
        if row is None:
            return None
        row["orders"] = json.loads(row.pop("orders_json"))
        return row

    def get_order_proposal_by_id(self, proposal_id: int) -> Optional[Dict[str, Any]]:
        row = self.query_one(
            """
            SELECT *
            FROM order_proposals
            WHERE proposal_id = ?
            """,
            (int(proposal_id),),
        )
        if row is None:
            return None
        row["orders"] = json.loads(row.pop("orders_json"))
        return row

    def update_order_proposal_status(
        self, proposal_id: int, status: str, blocked_reason: str, updated_at: str
    ) -> None:
        with self.tx():
            self.execute(
                """
                UPDATE order_proposals
                SET status = ?, blocked_reason = ?, updated_at = ?
                WHERE proposal_id = ?
                """,
                (status, blocked_reason, updated_at, int(proposal_id)),
            )

    def bulk_update_pending_proposals(
        self,
        *,
        new_status: str,
        blocked_reason: str,
        updated_at: str,
    ) -> int:
        with self.tx():
            cur = self.execute(
                """
                UPDATE order_proposals
                SET status = ?, blocked_reason = ?, updated_at = ?
                WHERE status = 'PENDING_APPROVAL'
                """,
                (new_status, blocked_reason, updated_at),
            )
            return int(cur.rowcount or 0)

    def list_order_proposals(
        self,
        market: Optional[str] = None,
        session_date: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where: List[str] = []
        params: List[Any] = []
        if market:
            where.append("market = ?")
            params.append(market)
        if session_date:
            where.append("session_date = ?")
            params.append(session_date)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = "SELECT * FROM order_proposals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, proposal_id DESC"
        rows = self.query_all(sql, tuple(params))
        for row in rows:
            row["orders"] = json.loads(row.pop("orders_json"))
        return rows

    def insert_order_approval(
        self,
        proposal_id: int,
        approved_by: str,
        approved_at: str,
        decision: str,
        note: str,
    ) -> int:
        with self.tx():
            cur = self.execute(
                """
                INSERT INTO order_approvals(proposal_id, approved_by, approved_at, decision, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (int(proposal_id), approved_by, approved_at, decision, note),
            )
            return int(cur.lastrowid)

    def insert_paper_order(
        self,
        proposal_id: int,
        agent_id: str,
        market: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        limit_price: Optional[float],
        reference_price: float,
        status: str,
        broker_order_id: str,
        broker_response_json: Dict[str, Any],
        submitted_at: str,
    ) -> int:
        with self.tx():
            cur = self.execute(
                """
                INSERT INTO paper_orders(
                    proposal_id, agent_id, market, symbol, side, order_type, quantity,
                    limit_price, reference_price, status, broker_order_id, broker_response_json, submitted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(proposal_id),
                    agent_id,
                    market,
                    symbol,
                    side,
                    order_type,
                    float(quantity),
                    None if limit_price is None else float(limit_price),
                    float(reference_price),
                    status,
                    broker_order_id,
                    json.dumps(broker_response_json, ensure_ascii=False),
                    submitted_at,
                ),
            )
            return int(cur.lastrowid)

    def update_paper_order_status(self, paper_order_id: int, status: str) -> None:
        with self.tx():
            self.execute(
                """
                UPDATE paper_orders
                SET status = ?
                WHERE paper_order_id = ?
                """,
                (status, int(paper_order_id)),
            )

    def insert_paper_fill(
        self,
        paper_order_id: int,
        fill_price: float,
        fill_quantity: float,
        fill_value_krw: float,
        fx_rate: float,
        filled_at: str,
    ) -> int:
        with self.tx():
            cur = self.execute(
                """
                INSERT INTO paper_fills(
                    paper_order_id, fill_price, fill_quantity, fill_value_krw, fx_rate, filled_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(paper_order_id),
                    float(fill_price),
                    float(fill_quantity),
                    float(fill_value_krw),
                    float(fx_rate),
                    filled_at,
                ),
            )
            return int(cur.lastrowid)

    def list_paper_fills(self, agent_id: str, from_date: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = """
            SELECT pf.*, po.agent_id, po.market, po.symbol, po.side, po.submitted_at
            FROM paper_fills pf
            JOIN paper_orders po ON po.paper_order_id = pf.paper_order_id
            WHERE po.agent_id = ?
        """
        params: List[Any] = [agent_id]
        if from_date:
            sql += " AND substr(pf.filled_at, 1, 10) >= ?"
            params.append(from_date)
        sql += " ORDER BY pf.filled_at ASC, pf.paper_fill_id ASC"
        return self.query_all(sql, tuple(params))

    def upsert_position(
        self,
        agent_id: str,
        market: str,
        symbol: str,
        quantity: float,
        avg_price: float,
        market_value_krw: float,
        unrealized_pnl_krw: float,
        updated_at: str,
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO positions(
                    agent_id, market, symbol, quantity, avg_price, market_value_krw, unrealized_pnl_krw, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, market, symbol) DO UPDATE SET
                    quantity=excluded.quantity,
                    avg_price=excluded.avg_price,
                    market_value_krw=excluded.market_value_krw,
                    unrealized_pnl_krw=excluded.unrealized_pnl_krw,
                    updated_at=excluded.updated_at
                """,
                (
                    agent_id,
                    market,
                    symbol,
                    float(quantity),
                    float(avg_price),
                    float(market_value_krw),
                    float(unrealized_pnl_krw),
                    updated_at,
                ),
            )

    def delete_agent_positions(self, agent_id: str) -> None:
        with self.tx():
            self.execute("DELETE FROM positions WHERE agent_id = ?", (agent_id,))

    def list_positions(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if agent_id:
            return self.query_all(
                "SELECT * FROM positions WHERE agent_id = ? ORDER BY agent_id, market, symbol",
                (agent_id,),
            )
        return self.query_all("SELECT * FROM positions ORDER BY agent_id, market, symbol")

    def upsert_equity_snapshot(
        self,
        agent_id: str,
        as_of_date: str,
        cash_krw: float,
        invested_krw: float,
        equity_krw: float,
        realized_pnl_krw: float,
        unrealized_pnl_krw: float,
        drawdown: float,
        volatility: float,
        win_rate: float,
        profit_factor: float,
        turnover: float,
        max_consecutive_loss: int,
        created_at: str,
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO equity_curve(
                    agent_id, as_of_date, cash_krw, invested_krw, equity_krw,
                    realized_pnl_krw, unrealized_pnl_krw, drawdown, volatility,
                    win_rate, profit_factor, turnover, max_consecutive_loss, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, as_of_date) DO UPDATE SET
                    cash_krw=excluded.cash_krw,
                    invested_krw=excluded.invested_krw,
                    equity_krw=excluded.equity_krw,
                    realized_pnl_krw=excluded.realized_pnl_krw,
                    unrealized_pnl_krw=excluded.unrealized_pnl_krw,
                    drawdown=excluded.drawdown,
                    volatility=excluded.volatility,
                    win_rate=excluded.win_rate,
                    profit_factor=excluded.profit_factor,
                    turnover=excluded.turnover,
                    max_consecutive_loss=excluded.max_consecutive_loss,
                    created_at=excluded.created_at
                """,
                (
                    agent_id,
                    as_of_date,
                    float(cash_krw),
                    float(invested_krw),
                    float(equity_krw),
                    float(realized_pnl_krw),
                    float(unrealized_pnl_krw),
                    float(drawdown),
                    float(volatility),
                    float(win_rate),
                    float(profit_factor),
                    float(turnover),
                    int(max_consecutive_loss),
                    created_at,
                ),
            )

    def list_equity_curve(
        self, agent_id: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        where: List[str] = []
        params: List[Any] = []
        if agent_id:
            where.append("agent_id = ?")
            params.append(agent_id)
        if date_from:
            where.append("as_of_date >= ?")
            params.append(date_from)
        if date_to:
            where.append("as_of_date <= ?")
            params.append(date_to)
        sql = "SELECT * FROM equity_curve"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY as_of_date ASC, agent_id ASC"
        return self.query_all(sql, tuple(params))

    def upsert_daily_review(
        self, review_date: str, summary: Dict[str, Any], markdown: str, created_at: str
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO daily_reviews(review_date, summary_json, markdown, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(review_date) DO UPDATE SET
                    summary_json=excluded.summary_json,
                    markdown=excluded.markdown,
                    created_at=excluded.created_at
                """,
                (review_date, json.dumps(summary, ensure_ascii=False), markdown, created_at),
            )

    def upsert_weekly_council(
        self,
        week_id: str,
        champion_agent_id: str,
        decision: Dict[str, Any],
        markdown: str,
        created_at: str,
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO weekly_councils(week_id, champion_agent_id, decision_json, markdown, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(week_id) DO UPDATE SET
                    champion_agent_id=excluded.champion_agent_id,
                    decision_json=excluded.decision_json,
                    markdown=excluded.markdown,
                    created_at=excluded.created_at
                """,
                (
                    week_id,
                    champion_agent_id,
                    json.dumps(decision, ensure_ascii=False),
                    markdown,
                    created_at,
                ),
            )

    def insert_agent_memory(
        self, agent_id: str, memory_type: str, content: Dict[str, Any], created_at: str
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO agent_memories(agent_id, memory_type, content_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (agent_id, memory_type, json.dumps(content, ensure_ascii=False), created_at),
            )

    def list_agent_memories(self, agent_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.query_all(
            """
            SELECT *
            FROM agent_memories
            WHERE agent_id = ?
            ORDER BY created_at DESC, memory_id DESC
            LIMIT ?
            """,
            (agent_id, int(limit)),
        )
        for row in rows:
            row["content"] = json.loads(row.pop("content_json"))
        return rows

    def delete_legacy_agent_memories(self, agent_id: str) -> int:
        with self.tx():
            cur = self.execute(
                """
                DELETE FROM agent_memories
                WHERE agent_id = ?
                  AND (
                    content_json LIKE '%PENDING_APPROVAL%'
                    OR content_json LIKE '%max_daily_picks=3%'
                    OR content_json LIKE '%exposure_cap_ratio=0.95%'
                    OR content_json LIKE '%collect_seconds=600%'
                    OR content_json LIKE '%scheduled_daily%'
                    OR content_json LIKE '%not always-on loop%'
                    OR content_json LIKE '%approval flow%'
                    OR content_json LIKE '%Hard Constraints%'
                  )
                """,
                (agent_id,),
            )
            return int(cur.rowcount or 0)

    def get_latest_event(self, event_type: str) -> Optional[Dict[str, Any]]:
        row = self.query_one(
            """
            SELECT *
            FROM state_events
            WHERE event_type = ?
            ORDER BY created_at DESC, event_id DESC
            LIMIT 1
            """,
            (event_type,),
        )
        if row is None:
            return None
        row["payload"] = json.loads(row.pop("payload_json"))
        return row

    def list_events(self, event_type: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if event_type:
            rows = self.query_all(
                """
                SELECT *
                FROM state_events
                WHERE event_type = ?
                ORDER BY created_at DESC, event_id DESC
                LIMIT ?
                """,
                (event_type, int(limit)),
            )
        else:
            rows = self.query_all(
                """
                SELECT *
                FROM state_events
                ORDER BY created_at DESC, event_id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
        for row in rows:
            row["payload"] = json.loads(row.pop("payload_json"))
        return rows

    def cleanup_legacy_runtime_state(self, *, retain_days: int = 30, keep_agent_memories: int = 300) -> Dict[str, int]:
        retain_days = max(1, int(retain_days))
        keep_agent_memories = max(50, int(keep_agent_memories))
        cutoff_dt = datetime.now() - timedelta(days=retain_days)
        cutoff_iso = cutoff_dt.isoformat(timespec="seconds")
        cutoff_date = cutoff_dt.strftime("%Y%m%d")

        legacy_payload_markers = [
            "%PENDING_APPROVAL%",
            "%approval flow%",
            "%scheduled_daily%",
            "%not always-on loop%",
            "%max_daily_picks=3%",
            "%exposure_cap_ratio=0.95%",
            "%collect_seconds=600%",
            "%Hard Constraints%",
        ]

        removed = {
            "order_approvals_removed": 0,
            "order_proposals_removed": 0,
            "session_signals_removed": 0,
            "state_events_removed": 0,
            "daily_reviews_removed": 0,
            "agent_memories_trimmed": 0,
        }

        with self.tx():
            proposal_rows = self.query_all(
                """
                SELECT op.proposal_id
                FROM order_proposals op
                WHERE (
                    op.status = 'PENDING_APPROVAL'
                    OR op.blocked_reason = 'legacy_cleanup_no_manual_approval'
                    OR (
                        op.status IN ('APPROVED', 'BLOCKED', 'REJECTED')
                        AND op.created_at < ?
                        AND NOT EXISTS (
                            SELECT 1
                            FROM paper_orders po
                            WHERE po.proposal_id = op.proposal_id
                        )
                    )
                )
                """,
                (cutoff_iso,),
            )
            proposal_ids = [int(row["proposal_id"]) for row in proposal_rows]
            if proposal_ids:
                placeholders = ",".join("?" for _ in proposal_ids)
                cur = self.execute(
                    f"DELETE FROM order_approvals WHERE proposal_id IN ({placeholders})",
                    tuple(proposal_ids),
                )
                removed["order_approvals_removed"] = int(cur.rowcount or 0)
                cur = self.execute(
                    f"DELETE FROM order_proposals WHERE proposal_id IN ({placeholders})",
                    tuple(proposal_ids),
                )
                removed["order_proposals_removed"] = int(cur.rowcount or 0)

            cur = self.execute(
                """
                DELETE FROM session_signals
                WHERE created_at < ?
                """,
                (cutoff_iso,),
            )
            removed["session_signals_removed"] = int(cur.rowcount or 0)

            where_markers = " OR ".join(["payload_json LIKE ?"] * len(legacy_payload_markers))
            cur = self.execute(
                f"""
                DELETE FROM state_events
                WHERE created_at < ?
                   OR ({where_markers})
                """,
                tuple([cutoff_iso] + legacy_payload_markers),
            )
            removed["state_events_removed"] = int(cur.rowcount or 0)

            cur = self.execute(
                """
                DELETE FROM daily_reviews
                WHERE review_date < ?
                """,
                (cutoff_date,),
            )
            removed["daily_reviews_removed"] = int(cur.rowcount or 0)

            agents = self.query_all("SELECT agent_id FROM agents WHERE is_active = 1")
            for agent in agents:
                aid = str(agent.get("agent_id", ""))
                if not aid:
                    continue
                stale_rows = self.query_all(
                    """
                    SELECT memory_id
                    FROM agent_memories
                    WHERE agent_id = ?
                    ORDER BY created_at DESC, memory_id DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (aid, keep_agent_memories),
                )
                stale_ids = [int(row["memory_id"]) for row in stale_rows]
                if stale_ids:
                    placeholders = ",".join("?" for _ in stale_ids)
                    cur = self.execute(
                        f"DELETE FROM agent_memories WHERE memory_id IN ({placeholders})",
                        tuple(stale_ids),
                    )
                    removed["agent_memories_trimmed"] += int(cur.rowcount or 0)

        return removed
