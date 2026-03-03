from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class TraderStorage:
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
                CREATE TABLE IF NOT EXISTS system_meta (
                    meta_key TEXT PRIMARY KEY,
                    meta_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS day_budget (
                    trade_date TEXT PRIMARY KEY,
                    day_start_cash_snapshot_total REAL NOT NULL,
                    per_trade_ratio REAL NOT NULL,
                    budget_per_trade REAL NOT NULL,
                    captured_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS premarket_calc_log (
                    premarket_calc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    market TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    universe_source TEXT NOT NULL,
                    candidate_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    detail_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_premarket_calc_date_market
                    ON premarket_calc_log(trade_date, market, premarket_calc_id DESC);

                CREATE TABLE IF NOT EXISTS market_filter (
                    trade_date TEXT NOT NULL,
                    market TEXT NOT NULL,
                    index_symbol TEXT NOT NULL,
                    prev_close REAL,
                    ma20_prev REAL,
                    trading_enabled INTEGER NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    computed_at TEXT NOT NULL,
                    PRIMARY KEY(trade_date, market)
                );

                CREATE TABLE IF NOT EXISTS symbol_plan (
                    trade_date TEXT NOT NULL,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    candidate_rank INTEGER NOT NULL DEFAULT 0,
                    prev_high REAL,
                    prev_low REAL,
                    today_open REAL,
                    breakout_price REAL,
                    entered_today INTEGER NOT NULL DEFAULT 0,
                    last_price REAL,
                    last_seen_at TEXT,
                    is_candidate INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(trade_date, market, symbol)
                );
                CREATE INDEX IF NOT EXISTS idx_symbol_plan_market_date
                    ON symbol_plan(market, trade_date, candidate_rank);

                CREATE TABLE IF NOT EXISTS account_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    market_scope TEXT NOT NULL,
                    source TEXT NOT NULL,
                    strict INTEGER NOT NULL DEFAULT 1,
                    ok INTEGER NOT NULL DEFAULT 0,
                    blocked INTEGER NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL DEFAULT '',
                    cash_krw REAL NOT NULL DEFAULT 0,
                    equity_krw REAL NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_account_snapshots_scope_ts
                    ON account_snapshots(market_scope, snapshot_id DESC);

                CREATE TABLE IF NOT EXISTS positions_server (
                    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id INTEGER NOT NULL,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL DEFAULT 0,
                    avg_price REAL NOT NULL DEFAULT 0,
                    market_value_krw REAL NOT NULL DEFAULT 0,
                    currency TEXT NOT NULL DEFAULT 'KRW',
                    fx_rate REAL NOT NULL DEFAULT 1,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(snapshot_id) REFERENCES account_snapshots(snapshot_id)
                );
                CREATE INDEX IF NOT EXISTS idx_positions_server_snapshot
                    ON positions_server(snapshot_id, market, symbol);

                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    market TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    reference_price REAL,
                    status TEXT NOT NULL,
                    reject_reason TEXT NOT NULL DEFAULT '',
                    broker_order_id TEXT NOT NULL DEFAULT '',
                    broker_response_json TEXT NOT NULL DEFAULT '{}',
                    submitted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_orders_date_market
                    ON orders(trade_date, market, order_id DESC);
                CREATE INDEX IF NOT EXISTS idx_orders_status
                    ON orders(status, submitted_at DESC);

                CREATE TABLE IF NOT EXISTS fills (
                    fill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER NOT NULL,
                    fill_price REAL NOT NULL,
                    fill_quantity REAL NOT NULL,
                    fill_value_krw REAL NOT NULL,
                    fx_rate REAL NOT NULL DEFAULT 1,
                    filled_at TEXT NOT NULL,
                    FOREIGN KEY(order_id) REFERENCES orders(order_id)
                );
                CREATE INDEX IF NOT EXISTS idx_fills_order
                    ON fills(order_id, fill_id DESC);

                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_type_ts
                    ON events(event_type, event_id DESC);

                CREATE TABLE IF NOT EXISTS daily_report (
                    trade_date TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def upsert_meta(self, key: str, value: Any) -> None:
        now = self.now_iso()
        payload = value
        if not isinstance(value, str):
            payload = json.dumps(value, ensure_ascii=False)
        with self.tx():
            self.execute(
                """
                INSERT INTO system_meta(meta_key, meta_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(meta_key) DO UPDATE SET
                    meta_value=excluded.meta_value,
                    updated_at=excluded.updated_at
                """,
                (str(key), str(payload), now),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT meta_value FROM system_meta WHERE meta_key = ?", (str(key),))
        if not row:
            return default
        raw = str(row.get("meta_value", ""))
        try:
            return json.loads(raw)
        except Exception:
            return raw if raw != "" else default

    def upsert_day_budget(
        self,
        *,
        trade_date: str,
        day_start_cash_snapshot_total: float,
        per_trade_ratio: float,
        budget_per_trade: float,
        captured_at: str,
        payload: Dict[str, Any],
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO day_budget(
                    trade_date, day_start_cash_snapshot_total, per_trade_ratio,
                    budget_per_trade, captured_at, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date) DO UPDATE SET
                    day_start_cash_snapshot_total=excluded.day_start_cash_snapshot_total,
                    per_trade_ratio=excluded.per_trade_ratio,
                    budget_per_trade=excluded.budget_per_trade,
                    captured_at=excluded.captured_at,
                    payload_json=excluded.payload_json
                """,
                (
                    str(trade_date),
                    float(day_start_cash_snapshot_total),
                    float(per_trade_ratio),
                    float(budget_per_trade),
                    str(captured_at),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def get_day_budget(self, trade_date: str) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM day_budget WHERE trade_date = ?", (str(trade_date),))
        if not row:
            return None
        try:
            row["payload"] = json.loads(str(row.pop("payload_json") or "{}"))
        except Exception:
            row["payload"] = {}
        return row

    def add_premarket_log(
        self,
        *,
        trade_date: str,
        market: str,
        started_at: str,
        finished_at: str | None,
        universe_source: str,
        candidate_count: int,
        status: str,
        detail: Dict[str, Any],
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO premarket_calc_log(
                    trade_date, market, started_at, finished_at,
                    universe_source, candidate_count, status, detail_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(trade_date),
                    str(market).upper(),
                    str(started_at),
                    None if finished_at is None else str(finished_at),
                    str(universe_source),
                    int(candidate_count),
                    str(status),
                    json.dumps(detail, ensure_ascii=False),
                ),
            )

    def upsert_market_filter(
        self,
        *,
        trade_date: str,
        market: str,
        index_symbol: str,
        prev_close: float | None,
        ma20_prev: float | None,
        trading_enabled: bool,
        reason: str,
        computed_at: str,
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO market_filter(
                    trade_date, market, index_symbol, prev_close, ma20_prev,
                    trading_enabled, reason, computed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_date, market) DO UPDATE SET
                    index_symbol=excluded.index_symbol,
                    prev_close=excluded.prev_close,
                    ma20_prev=excluded.ma20_prev,
                    trading_enabled=excluded.trading_enabled,
                    reason=excluded.reason,
                    computed_at=excluded.computed_at
                """,
                (
                    str(trade_date),
                    str(market).upper(),
                    str(index_symbol),
                    None if prev_close is None else float(prev_close),
                    None if ma20_prev is None else float(ma20_prev),
                    1 if trading_enabled else 0,
                    str(reason),
                    str(computed_at),
                ),
            )

    def get_market_filter(self, trade_date: str, market: str) -> Optional[Dict[str, Any]]:
        row = self.query_one(
            "SELECT * FROM market_filter WHERE trade_date = ? AND market = ?",
            (str(trade_date), str(market).upper()),
        )
        if not row:
            return None
        row["trading_enabled"] = bool(int(row.get("trading_enabled", 0) or 0))
        return row

    def upsert_symbol_plan_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        now = self.now_iso()
        with self.tx():
            for row in rows:
                self.execute(
                    """
                    INSERT INTO symbol_plan(
                        trade_date, market, symbol, name, candidate_rank,
                        prev_high, prev_low, today_open, breakout_price,
                        entered_today, last_price, last_seen_at, is_candidate
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trade_date, market, symbol) DO UPDATE SET
                        name=excluded.name,
                        candidate_rank=excluded.candidate_rank,
                        prev_high=excluded.prev_high,
                        prev_low=excluded.prev_low,
                        today_open=COALESCE(excluded.today_open, symbol_plan.today_open),
                        breakout_price=COALESCE(excluded.breakout_price, symbol_plan.breakout_price),
                        is_candidate=excluded.is_candidate,
                        last_price=COALESCE(excluded.last_price, symbol_plan.last_price),
                        last_seen_at=COALESCE(excluded.last_seen_at, symbol_plan.last_seen_at)
                    """,
                    (
                        str(row.get("trade_date", "")),
                        str(row.get("market", "")).upper(),
                        str(row.get("symbol", "")).upper(),
                        str(row.get("name", "")),
                        int(row.get("candidate_rank", 0) or 0),
                        row.get("prev_high"),
                        row.get("prev_low"),
                        row.get("today_open"),
                        row.get("breakout_price"),
                        1 if bool(row.get("entered_today", False)) else 0,
                        row.get("last_price"),
                        str(row.get("last_seen_at") or now),
                        1 if bool(row.get("is_candidate", True)) else 0,
                    ),
                )

    def clear_candidate_flags(self, trade_date: str, market: str) -> None:
        with self.tx():
            self.execute(
                """
                UPDATE symbol_plan
                SET is_candidate = 0, candidate_rank = 0, last_seen_at = ?
                WHERE trade_date = ? AND market = ?
                """,
                (self.now_iso(), str(trade_date), str(market).upper()),
            )

    def mark_entered_today(self, *, trade_date: str, market: str, symbol: str, entered: bool = True) -> None:
        with self.tx():
            self.execute(
                """
                UPDATE symbol_plan
                SET entered_today = ?, last_seen_at = ?
                WHERE trade_date = ? AND market = ? AND symbol = ?
                """,
                (
                    1 if entered else 0,
                    self.now_iso(),
                    str(trade_date),
                    str(market).upper(),
                    str(symbol).upper(),
                ),
            )

    def update_symbol_snapshot(
        self,
        *,
        trade_date: str,
        market: str,
        symbol: str,
        last_price: float | None,
        today_open: float | None,
        breakout_k: float,
    ) -> None:
        with self.tx():
            self.execute(
                """
                UPDATE symbol_plan
                SET
                    last_price = COALESCE(?, last_price),
                    today_open = COALESCE(?, today_open),
                    breakout_price = CASE
                        WHEN COALESCE(?, today_open) IS NOT NULL AND prev_high IS NOT NULL AND prev_low IS NOT NULL
                        THEN COALESCE(?, today_open) + (prev_high - prev_low) * ?
                        ELSE breakout_price
                    END,
                    last_seen_at = ?
                WHERE trade_date = ? AND market = ? AND symbol = ?
                """,
                (
                    None if last_price is None else float(last_price),
                    None if today_open is None else float(today_open),
                    None if today_open is None else float(today_open),
                    None if today_open is None else float(today_open),
                    float(breakout_k),
                    self.now_iso(),
                    str(trade_date),
                    str(market).upper(),
                    str(symbol).upper(),
                ),
            )

    def list_candidate_symbols(self, trade_date: str, market: str) -> List[Dict[str, Any]]:
        rows = self.query_all(
            """
            SELECT *
            FROM symbol_plan
            WHERE trade_date = ? AND market = ? AND is_candidate = 1
            ORDER BY candidate_rank ASC, symbol ASC
            """,
            (str(trade_date), str(market).upper()),
        )
        for row in rows:
            row["entered_today"] = bool(int(row.get("entered_today", 0) or 0))
        return rows

    def list_symbol_plan(self, trade_date: str, market: str) -> List[Dict[str, Any]]:
        rows = self.query_all(
            """
            SELECT *
            FROM symbol_plan
            WHERE trade_date = ? AND market = ?
            ORDER BY is_candidate DESC, candidate_rank ASC, symbol ASC
            """,
            (str(trade_date), str(market).upper()),
        )
        for row in rows:
            row["entered_today"] = bool(int(row.get("entered_today", 0) or 0))
        return rows

    def insert_account_snapshot(
        self,
        *,
        trade_date: str,
        market_scope: str,
        source: str,
        strict: bool,
        ok: bool,
        blocked: bool,
        reason: str,
        cash_krw: float,
        equity_krw: float,
        payload: Dict[str, Any],
        positions: Iterable[Dict[str, Any]],
        created_at: str,
    ) -> int:
        with self.tx():
            cur = self.execute(
                """
                INSERT INTO account_snapshots(
                    trade_date, market_scope, source, strict, ok, blocked,
                    reason, cash_krw, equity_krw, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(trade_date),
                    str(market_scope).upper(),
                    str(source),
                    1 if strict else 0,
                    1 if ok else 0,
                    1 if blocked else 0,
                    str(reason),
                    float(cash_krw),
                    float(equity_krw),
                    json.dumps(payload, ensure_ascii=False),
                    str(created_at),
                ),
            )
            snapshot_id = int(cur.lastrowid)
            for pos in positions:
                self.execute(
                    """
                    INSERT INTO positions_server(
                        snapshot_id, market, symbol, quantity, avg_price,
                        market_value_krw, currency, fx_rate, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(pos.get("market", "")).upper(),
                        str(pos.get("symbol", "")).upper(),
                        float(pos.get("quantity", 0.0) or 0.0),
                        float(pos.get("avg_price", 0.0) or 0.0),
                        float(pos.get("market_value_krw", 0.0) or 0.0),
                        str(pos.get("currency", "KRW") or "KRW").upper(),
                        float(pos.get("fx_rate", 1.0) or 1.0),
                        json.dumps(pos, ensure_ascii=False),
                    ),
                )
            return snapshot_id

    def latest_account_snapshot(self, market_scope: str = "ALL") -> Optional[Dict[str, Any]]:
        row = self.query_one(
            """
            SELECT *
            FROM account_snapshots
            WHERE market_scope = ?
            ORDER BY snapshot_id DESC
            LIMIT 1
            """,
            (str(market_scope).upper(),),
        )
        if not row:
            return None
        row["ok"] = bool(int(row.get("ok", 0) or 0))
        row["blocked"] = bool(int(row.get("blocked", 0) or 0))
        row["strict"] = bool(int(row.get("strict", 0) or 0))
        try:
            row["payload"] = json.loads(str(row.pop("payload_json") or "{}"))
        except Exception:
            row["payload"] = {}
        return row

    def list_positions_for_snapshot(self, snapshot_id: int, market: str | None = None) -> List[Dict[str, Any]]:
        if market is None:
            rows = self.query_all(
                "SELECT * FROM positions_server WHERE snapshot_id = ? ORDER BY market, symbol",
                (int(snapshot_id),),
            )
        else:
            rows = self.query_all(
                """
                SELECT *
                FROM positions_server
                WHERE snapshot_id = ? AND market = ?
                ORDER BY symbol
                """,
                (int(snapshot_id), str(market).upper()),
            )
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                row["payload"] = json.loads(str(row.pop("payload_json") or "{}"))
            except Exception:
                row["payload"] = {}
            out.append(row)
        return out

    def latest_positions(self, market_scope: str = "ALL", market: str | None = None) -> List[Dict[str, Any]]:
        latest = self.latest_account_snapshot(market_scope=market_scope)
        if not latest:
            return []
        return self.list_positions_for_snapshot(int(latest["snapshot_id"]), market=market)

    def insert_order(
        self,
        *,
        trade_date: str,
        market: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        reference_price: float | None,
        status: str,
        reject_reason: str,
        broker_order_id: str,
        broker_response: Dict[str, Any],
        submitted_at: str,
    ) -> int:
        with self.tx():
            cur = self.execute(
                """
                INSERT INTO orders(
                    trade_date, market, symbol, side, order_type,
                    quantity, reference_price, status, reject_reason,
                    broker_order_id, broker_response_json, submitted_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(trade_date),
                    str(market).upper(),
                    str(symbol).upper(),
                    str(side).upper(),
                    str(order_type).upper(),
                    float(quantity),
                    None if reference_price is None else float(reference_price),
                    str(status).upper(),
                    str(reject_reason),
                    str(broker_order_id),
                    json.dumps(broker_response, ensure_ascii=False),
                    str(submitted_at),
                    str(submitted_at),
                ),
            )
            return int(cur.lastrowid)

    def update_order_status(self, order_id: int, status: str, reject_reason: str = "") -> None:
        with self.tx():
            self.execute(
                """
                UPDATE orders
                SET status = ?, reject_reason = ?, updated_at = ?
                WHERE order_id = ?
                """,
                (str(status).upper(), str(reject_reason), self.now_iso(), int(order_id)),
            )

    def insert_fill(
        self,
        *,
        order_id: int,
        fill_price: float,
        fill_quantity: float,
        fill_value_krw: float,
        fx_rate: float,
        filled_at: str,
    ) -> None:
        with self.tx():
            self.execute(
                """
                INSERT INTO fills(order_id, fill_price, fill_quantity, fill_value_krw, fx_rate, filled_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(order_id),
                    float(fill_price),
                    float(fill_quantity),
                    float(fill_value_krw),
                    float(fx_rate),
                    str(filled_at),
                ),
            )

    def list_orders(self, trade_date: str, market: str | None = None) -> List[Dict[str, Any]]:
        if market is None:
            rows = self.query_all(
                "SELECT * FROM orders WHERE trade_date = ? ORDER BY order_id DESC",
                (str(trade_date),),
            )
        else:
            rows = self.query_all(
                "SELECT * FROM orders WHERE trade_date = ? AND market = ? ORDER BY order_id DESC",
                (str(trade_date), str(market).upper()),
            )
        for row in rows:
            try:
                row["broker_response"] = json.loads(str(row.pop("broker_response_json") or "{}"))
            except Exception:
                row["broker_response"] = {}
        return rows

    def list_fills(self, trade_date: str, market: str | None = None) -> List[Dict[str, Any]]:
        if market is None:
            rows = self.query_all(
                """
                SELECT f.*, o.market, o.symbol, o.side
                FROM fills f
                JOIN orders o ON o.order_id = f.order_id
                WHERE o.trade_date = ?
                ORDER BY f.fill_id DESC
                """,
                (str(trade_date),),
            )
        else:
            rows = self.query_all(
                """
                SELECT f.*, o.market, o.symbol, o.side
                FROM fills f
                JOIN orders o ON o.order_id = f.order_id
                WHERE o.trade_date = ? AND o.market = ?
                ORDER BY f.fill_id DESC
                """,
                (str(trade_date), str(market).upper()),
            )
        return rows

    def log_event(self, event_type: str, payload: Dict[str, Any], created_at: str | None = None) -> None:
        ts = str(created_at or self.now_iso())
        with self.tx():
            self.execute(
                "INSERT INTO events(event_type, payload_json, created_at) VALUES (?, ?, ?)",
                (str(event_type), json.dumps(payload, ensure_ascii=False), ts),
            )

    def list_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.query_all(
            "SELECT * FROM events ORDER BY event_id DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        for row in rows:
            try:
                row["payload"] = json.loads(str(row.pop("payload_json") or "{}"))
            except Exception:
                row["payload"] = {}
        return rows

    def upsert_daily_report(self, trade_date: str, payload: Dict[str, Any]) -> None:
        now = self.now_iso()
        with self.tx():
            self.execute(
                """
                INSERT INTO daily_report(trade_date, payload_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(trade_date) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    created_at=excluded.created_at
                """,
                (str(trade_date), json.dumps(payload, ensure_ascii=False), now),
            )

    def get_daily_report(self, trade_date: str) -> Optional[Dict[str, Any]]:
        row = self.query_one("SELECT * FROM daily_report WHERE trade_date = ?", (str(trade_date),))
        if not row:
            return None
        try:
            row["payload"] = json.loads(str(row.pop("payload_json") or "{}"))
        except Exception:
            row["payload"] = {}
        return row

    def acquire_lock(self, key: str, owner: str) -> bool:
        now = self.now_iso()
        with self.tx():
            cur = self.execute(
                """
                INSERT OR IGNORE INTO system_meta(meta_key, meta_value, updated_at)
                VALUES (?, ?, ?)
                """,
                (f"lock:{key}", json.dumps({"owner": owner}, ensure_ascii=False), now),
            )
            return int(cur.rowcount or 0) > 0

    def release_lock(self, key: str) -> None:
        with self.tx():
            self.execute("DELETE FROM system_meta WHERE meta_key = ?", (f"lock:{key}",))
