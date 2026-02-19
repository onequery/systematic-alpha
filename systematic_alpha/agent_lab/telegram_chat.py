from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from systematic_alpha.agent_lab.identity import AgentIdentityStore
from systematic_alpha.agent_lab.llm_client import LLMClient
from systematic_alpha.agent_lab.storage import AgentLabStorage
from systematic_alpha.agent_lab.strategy_registry import ALLOWED_PARAM_RANGES, StrategyRegistry


def _truthy(value: str) -> bool:
    norm = str(value or "").strip().lower()
    return norm in {"1", "true", "yes", "y", "on"}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_token(token: str) -> str:
    clean = str(token or "").strip()
    if clean.lower().startswith("bot"):
        return clean[3:]
    return clean


def _truncate(text: str, max_chars: int = 3500) -> str:
    body = str(text or "")
    if len(body) <= max_chars:
        return body
    return body[:max_chars] + "\n...(truncated)..."


class TelegramChatRuntime:
    DIRECTIVE_EVENT_TYPE = "telegram_directive"
    DIRECTIVE_STATUS_PENDING = "PENDING"
    DIRECTIVE_STATUS_APPLIED = "APPLIED"
    DIRECTIVE_STATUS_REJECTED = "REJECTED"

    def __init__(
        self,
        *,
        project_root: str | Path,
        poll_timeout: int = 25,
        idle_sleep: float = 1.0,
        memory_limit: int = 20,
    ):
        self.project_root = Path(project_root).resolve()
        self.poll_timeout = max(1, int(poll_timeout))
        self.idle_sleep = max(0.1, float(idle_sleep))
        self.memory_limit = max(5, int(memory_limit))

        self.state_root = self.project_root / "state" / "agent_lab"
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.offset_path = self.state_root / "runtime" / "telegram_offset.txt"
        self.offset_path.parent.mkdir(parents=True, exist_ok=True)

        self.storage = AgentLabStorage(self.state_root / "agent_lab.sqlite")
        self.registry = StrategyRegistry(self.storage)
        self.identity = AgentIdentityStore(self.state_root, self.storage)
        self.llm = LLMClient(self.storage)

        token = _normalize_token(os.getenv("TELEGRAM_BOT_TOKEN", ""))
        chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        thread_id = str(os.getenv("TELEGRAM_THREAD_ID", "")).strip()
        enabled_raw = str(os.getenv("TELEGRAM_ENABLED", "")).strip()
        self.disable_notification = _truthy(os.getenv("TELEGRAM_DISABLE_NOTIFICATION", "0"))

        if enabled_raw:
            self.telegram_enabled = _truthy(enabled_raw) and bool(token and chat_id)
        else:
            self.telegram_enabled = bool(token and chat_id)

        self.token = token
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.session = requests.Session()
        if not _truthy(os.getenv("AGENT_LAB_TELEGRAM_USE_ENV_PROXY", "0")):
            self.session.trust_env = False
        self.offset = self._load_offset()
        self.processed_updates = 0

        self.agent_alias: Dict[str, str] = {
            "a": "agent_a",
            "b": "agent_b",
            "c": "agent_c",
            "agent-a": "agent_a",
            "agent-b": "agent_b",
            "agent-c": "agent_c",
            "agent_a": "agent_a",
            "agent_b": "agent_b",
            "agent_c": "agent_c",
        }

    def _sanitize_error(self, value: Any) -> str:
        text = repr(value)
        if self.token:
            text = text.replace(self.token, "***")
        return text

    def close(self) -> None:
        self.storage.close()

    def _load_offset(self) -> int:
        if not self.offset_path.exists():
            return 0
        try:
            return int(self.offset_path.read_text(encoding="utf-8").strip() or 0)
        except Exception:
            return 0

    def _save_offset(self, value: int) -> None:
        self.offset = int(value)
        self.offset_path.write_text(str(self.offset), encoding="utf-8")

    def _api_get(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/{method}", params=params, timeout=self.poll_timeout + 10)
        resp.raise_for_status()
        data = resp.json()
        if not bool(data.get("ok")):
            raise RuntimeError(f"telegram api error: {data.get('error_code')} {data.get('description')}")
        return data

    def _api_post(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/{method}",
            data=payload,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if not bool(data.get("ok")):
            raise RuntimeError(f"telegram api error: {data.get('error_code')} {data.get('description')}")
        return data

    def _send_message(self, text: str, *, chat_id: Optional[str] = None, thread_id: Optional[str] = None) -> None:
        if not self.telegram_enabled:
            return
        payload: Dict[str, Any] = {
            "chat_id": chat_id or self.chat_id,
            "text": _truncate(text),
            "disable_web_page_preview": "true",
        }
        if self.disable_notification:
            payload["disable_notification"] = "true"
        target_thread = thread_id if thread_id is not None else self.thread_id
        if str(target_thread or "").strip():
            payload["message_thread_id"] = str(target_thread).strip()
        try:
            self._api_post("sendMessage", payload)
        except Exception as exc:
            self.storage.log_event(
                event_type="telegram_chat_send_error",
                payload={"error": self._sanitize_error(exc)},
                created_at=_now_iso(),
            )

    def _parse_command(self, raw_text: str) -> Tuple[str, List[str]]:
        txt = str(raw_text or "").strip()
        if not txt:
            return "", []
        if not txt.startswith("/"):
            return "", txt.split()
        parts = txt.split()
        cmd = parts[0]
        if "@" in cmd:
            cmd = cmd.split("@", 1)[0]
        return cmd.lower(), parts[1:]

    def _normalize_agent_id(self, token: str) -> Optional[str]:
        key = str(token or "").strip().lower()
        if not key:
            return None
        if key in self.agent_alias:
            return self.agent_alias[key]
        agents = self.storage.list_agents()
        for row in agents:
            aid = str(row.get("agent_id", "")).strip()
            name = str(row.get("name", "")).strip().lower()
            if key == aid.lower():
                return aid
            if key == name:
                return aid
        return None

    @staticmethod
    def _as_int_param_key(key: str) -> bool:
        return key in {"min_pass_conditions", "collect_seconds"}

    def _actor_from_message(self, message: Dict[str, Any]) -> str:
        user = message.get("from") or {}
        uid = str(user.get("id", "")).strip()
        username = str(user.get("username", "")).strip()
        first = str(user.get("first_name", "")).strip()
        last = str(user.get("last_name", "")).strip()
        if username:
            return f"telegram:{uid}/{username}"
        name = " ".join([x for x in [first, last] if x]).strip()
        if name:
            return f"telegram:{uid}/{name}"
        return f"telegram:{uid}" if uid else "telegram:unknown"

    def _insert_directive(self, payload: Dict[str, Any]) -> int:
        created_at = _now_iso()
        with self.storage.tx():
            cur = self.storage.execute(
                """
                INSERT INTO state_events(event_type, payload_json, created_at)
                VALUES (?, ?, ?)
                """,
                (self.DIRECTIVE_EVENT_TYPE, json.dumps(payload, ensure_ascii=False), created_at),
            )
            return int(cur.lastrowid)

    def _directive_by_id(self, directive_id: int) -> Optional[Dict[str, Any]]:
        row = self.storage.query_one(
            """
            SELECT event_id, payload_json, created_at
            FROM state_events
            WHERE event_type = ? AND event_id = ?
            LIMIT 1
            """,
            (self.DIRECTIVE_EVENT_TYPE, int(directive_id)),
        )
        if row is None:
            return None
        payload: Dict[str, Any] = {}
        try:
            decoded = json.loads(str(row.get("payload_json") or "{}"))
            if isinstance(decoded, dict):
                payload = decoded
        except Exception:
            payload = {}
        payload["directive_id"] = int(row["event_id"])
        payload["created_at"] = str(row.get("created_at", ""))
        return payload

    def _update_directive(self, directive_id: int, payload: Dict[str, Any]) -> None:
        with self.storage.tx():
            self.storage.execute(
                """
                UPDATE state_events
                SET payload_json = ?
                WHERE event_type = ? AND event_id = ?
                """,
                (
                    json.dumps(payload, ensure_ascii=False),
                    self.DIRECTIVE_EVENT_TYPE,
                    int(directive_id),
                ),
            )

    def _list_directives(
        self,
        *,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        rows = self.storage.query_all(
            """
            SELECT event_id, payload_json, created_at
            FROM state_events
            WHERE event_type = ?
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (self.DIRECTIVE_EVENT_TYPE, int(limit)),
        )
        out: List[Dict[str, Any]] = []
        status_norm = str(status or "").strip().upper()
        for row in rows:
            payload: Dict[str, Any] = {}
            try:
                decoded = json.loads(str(row.get("payload_json") or "{}"))
                if isinstance(decoded, dict):
                    payload = decoded
            except Exception:
                payload = {}
            payload["directive_id"] = int(row["event_id"])
            payload["created_at"] = str(row.get("created_at", ""))
            aid = str(payload.get("agent_id", "")).strip()
            st = str(payload.get("status", "")).strip().upper()
            if agent_id and aid != agent_id:
                continue
            if status_norm and st != status_norm:
                continue
            out.append(payload)
        return out

    def _create_freeform_directive(self, agent_id: str, request_text: str, requested_by: str) -> int:
        payload = {
            "directive_type": "freeform",
            "agent_id": agent_id,
            "request_text": str(request_text or "").strip(),
            "status": self.DIRECTIVE_STATUS_PENDING,
            "requested_by": requested_by,
            "requested_at": _now_iso(),
            "decision": {},
            "applied": {},
        }
        did = self._insert_directive(payload)
        self.identity.append_memory(
            agent_id=agent_id,
            memory_type="operator_directive_requested",
            content={
                "directive_id": did,
                "directive_type": payload["directive_type"],
                "request_text": payload["request_text"],
                "requested_by": requested_by,
            },
            ts=_now_iso(),
        )
        return did

    def _coerce_param_value(self, key: str, raw: str) -> Tuple[bool, Optional[Any], str]:
        text = str(raw or "").strip()
        if key not in ALLOWED_PARAM_RANGES:
            keys = ", ".join(sorted(ALLOWED_PARAM_RANGES.keys()))
            return False, None, f"unknown param key: {key}. allowed: {keys}"
        try:
            num = float(text)
        except Exception:
            return False, None, f"param value must be numeric: {text}"
        if self._as_int_param_key(key):
            return True, int(round(num)), ""
        return True, float(num), ""

    def _create_param_directive(
        self,
        agent_id: str,
        param_key: str,
        param_value_raw: str,
        requested_by: str,
        note: str,
    ) -> Tuple[bool, str]:
        ok, value, err = self._coerce_param_value(param_key, param_value_raw)
        if not ok:
            return False, err
        payload = {
            "directive_type": "param_update",
            "agent_id": agent_id,
            "status": self.DIRECTIVE_STATUS_PENDING,
            "requested_by": requested_by,
            "requested_at": _now_iso(),
            "request_text": f"set {param_key}={param_value_raw}",
            "param_key": param_key,
            "param_value_raw": param_value_raw,
            "param_value": value,
            "note": str(note or "").strip(),
            "decision": {},
            "applied": {},
        }
        did = self._insert_directive(payload)
        self.identity.append_memory(
            agent_id=agent_id,
            memory_type="operator_directive_requested",
            content={
                "directive_id": did,
                "directive_type": payload["directive_type"],
                "param_key": param_key,
                "param_value_raw": param_value_raw,
                "requested_by": requested_by,
                "note": payload["note"],
            },
            ts=_now_iso(),
        )
        return True, f"directive #{did} created: {agent_id} set {param_key}={param_value_raw} (PENDING)"

    def _apply_directive(self, directive: Dict[str, Any], actor: str, note: str) -> Tuple[bool, str]:
        did = int(directive.get("directive_id", 0) or 0)
        if did <= 0:
            return False, "invalid directive id"
        agent_id = str(directive.get("agent_id", "")).strip()
        if not agent_id:
            return False, f"directive #{did} missing agent_id"

        if str(directive.get("status", "")).strip().upper() != self.DIRECTIVE_STATUS_PENDING:
            return False, f"directive #{did} is not pending"

        dtype = str(directive.get("directive_type", "")).strip()
        decision_note = str(note or "").strip()
        updated = dict(directive)
        updated["status"] = self.DIRECTIVE_STATUS_APPLIED
        updated["decision"] = {
            "decision": "APPROVED",
            "decided_by": actor,
            "decided_at": _now_iso(),
            "note": decision_note,
        }
        updated["applied"] = {"applied_at": _now_iso(), "applied_by": actor}

        if dtype == "param_update":
            param_key = str(directive.get("param_key", "")).strip()
            ok, value, err = self._coerce_param_value(param_key, str(directive.get("param_value_raw", "")))
            if not ok:
                return False, f"directive #{did} invalid param value: {err}"
            active = self.registry.get_active_strategy(agent_id)
            base_params = dict(active.get("params", {}))
            prev_value = base_params.get(param_key)
            base_params[param_key] = value
            reg = self.registry.register_strategy_version(
                agent_id=agent_id,
                params=base_params,
                notes=(
                    f"telegram directive #{did} approved by {actor}; "
                    f"{param_key}={value}; note={decision_note}"
                ),
                promote=True,
            )
            clamped_params = dict(reg.get("params", {}))
            applied_value = clamped_params.get(param_key)
            updated["applied"].update(
                {
                    "strategy_version_id": reg.get("strategy_version_id"),
                    "version_tag": reg.get("version_tag"),
                    "param_key": param_key,
                    "previous_value": prev_value,
                    "requested_value": value,
                    "applied_value": applied_value,
                }
            )
            self.identity.append_memory(
                agent_id=agent_id,
                memory_type="operator_directive_applied",
                content={
                    "directive_id": did,
                    "directive_type": dtype,
                    "param_key": param_key,
                    "previous_value": prev_value,
                    "requested_value": value,
                    "applied_value": applied_value,
                    "version_tag": reg.get("version_tag"),
                    "approved_by": actor,
                    "note": decision_note,
                },
                ts=_now_iso(),
            )
            self._update_directive(did, updated)
            self.storage.log_event(
                event_type="telegram_directive_applied",
                payload={
                    "directive_id": did,
                    "agent_id": agent_id,
                    "directive_type": dtype,
                    "param_key": param_key,
                    "requested_value": value,
                    "applied_value": applied_value,
                    "version_tag": reg.get("version_tag"),
                    "approved_by": actor,
                },
                created_at=_now_iso(),
            )
            return (
                True,
                (
                    f"directive #{did} APPLIED\n"
                    f"agent={agent_id}\n"
                    f"param={param_key}\n"
                    f"prev={prev_value}\n"
                    f"requested={value}\n"
                    f"applied={applied_value}\n"
                    f"strategy={reg.get('version_tag')}"
                ),
            )

        request_text = str(directive.get("request_text", "")).strip()
        self.identity.append_memory(
            agent_id=agent_id,
            memory_type="operator_directive_applied",
            content={
                "directive_id": did,
                "directive_type": dtype or "freeform",
                "request_text": request_text,
                "approved_by": actor,
                "note": decision_note,
            },
            ts=_now_iso(),
        )
        self._update_directive(did, updated)
        self.storage.log_event(
            event_type="telegram_directive_applied",
            payload={
                "directive_id": did,
                "agent_id": agent_id,
                "directive_type": dtype or "freeform",
                "request_text": request_text,
                "approved_by": actor,
            },
            created_at=_now_iso(),
        )
        return (
            True,
            (
                f"directive #{did} APPLIED\n"
                f"agent={agent_id}\n"
                f"type={dtype or 'freeform'}\n"
                f"request={request_text or '(empty)'}"
            ),
        )

    def _reject_directive(self, directive: Dict[str, Any], actor: str, note: str) -> Tuple[bool, str]:
        did = int(directive.get("directive_id", 0) or 0)
        if did <= 0:
            return False, "invalid directive id"
        if str(directive.get("status", "")).strip().upper() != self.DIRECTIVE_STATUS_PENDING:
            return False, f"directive #{did} is not pending"
        agent_id = str(directive.get("agent_id", "")).strip()
        updated = dict(directive)
        updated["status"] = self.DIRECTIVE_STATUS_REJECTED
        updated["decision"] = {
            "decision": "REJECTED",
            "decided_by": actor,
            "decided_at": _now_iso(),
            "note": str(note or "").strip(),
        }
        updated["applied"] = {}
        self._update_directive(did, updated)
        if agent_id:
            self.identity.append_memory(
                agent_id=agent_id,
                memory_type="operator_directive_rejected",
                content={
                    "directive_id": did,
                    "directive_type": str(directive.get("directive_type", "")),
                    "request_text": str(directive.get("request_text", "")),
                    "rejected_by": actor,
                    "note": str(note or "").strip(),
                },
                ts=_now_iso(),
            )
        self.storage.log_event(
            event_type="telegram_directive_rejected",
            payload={
                "directive_id": did,
                "agent_id": agent_id,
                "directive_type": str(directive.get("directive_type", "")),
                "rejected_by": actor,
                "note": str(note or "").strip(),
            },
            created_at=_now_iso(),
        )
        return True, f"directive #{did} REJECTED"

    def _latest_proposal(self, agent_id: str) -> Optional[Dict[str, Any]]:
        row = self.storage.query_one(
            """
            SELECT *
            FROM order_proposals
            WHERE agent_id = ?
            ORDER BY created_at DESC, proposal_id DESC
            LIMIT 1
            """,
            (agent_id,),
        )
        if row is None:
            return None
        try:
            row["orders"] = json.loads(row.pop("orders_json"))
        except Exception:
            row["orders"] = []
        return row

    def _latest_equity(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self.storage.query_one(
            """
            SELECT *
            FROM equity_curve
            WHERE agent_id = ?
            ORDER BY as_of_date DESC, equity_id DESC
            LIMIT 1
            """,
            (agent_id,),
        )

    def _latest_weekly_council(self) -> Optional[Dict[str, Any]]:
        row = self.storage.query_one(
            """
            SELECT *
            FROM weekly_councils
            ORDER BY created_at DESC, weekly_council_id DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        try:
            row["decision"] = json.loads(row.pop("decision_json"))
        except Exception:
            row["decision"] = {}
        return row

    def _build_agent_context(self, agent_id: str) -> Dict[str, Any]:
        warm = self.identity.build_warm_start_context(agent_id, memory_limit=self.memory_limit)
        active = self.registry.get_active_strategy(agent_id)
        latest_proposal = self._latest_proposal(agent_id)
        latest_equity = self._latest_equity(agent_id)
        recent_directives = self._list_directives(agent_id=agent_id, limit=8)
        positions = self.storage.list_positions(agent_id)
        if len(positions) > 20:
            positions = positions[:20]

        return {
            "agent_id": agent_id,
            "identity_markdown": warm.get("identity_markdown", ""),
            "recent_memories": warm.get("recent_memories", []),
            "latest_checkpoint": warm.get("latest_checkpoint"),
            "active_strategy": active,
            "latest_proposal": latest_proposal,
            "latest_equity": latest_equity,
            "recent_directives": recent_directives,
            "positions": positions,
        }

    def _fallback_answer(self, agent_id: str, question: str, ctx: Dict[str, Any]) -> str:
        proposal = ctx.get("latest_proposal") or {}
        orders = proposal.get("orders") or []
        order_chunks: List[str] = []
        for row in orders[:3]:
            side = str(row.get("side", ""))
            symbol = str(row.get("symbol", ""))
            qty = float(row.get("quantity", 0.0) or 0.0)
            order_chunks.append(f"{side} {symbol} x{qty:.0f}")
        orders_text = ", ".join(order_chunks) if order_chunks else "none"
        eq = ctx.get("latest_equity") or {}
        equity_krw = float(eq.get("equity_krw", 0.0) or 0.0)
        drawdown = float(eq.get("drawdown", 0.0) or 0.0)
        return (
            f"{agent_id} status snapshot:\n"
            f"- last_proposal_status: {proposal.get('status', 'N/A')}\n"
            f"- last_proposal_orders: {orders_text}\n"
            f"- equity_krw: {equity_krw:.0f}\n"
            f"- drawdown: {drawdown:.4f}\n"
            f"- question: {question}\n"
            "LLM is unavailable or budget-limited, so this is a deterministic status response."
        )

    def _is_token_budget_issue(self, reason: str) -> bool:
        key = str(reason or "").lower()
        markers = [
            "daily_budget_exceeded",
            "insufficient_quota",
            "quota",
            "billing",
            "context_length_exceeded",
            "max_tokens",
            "token",
        ]
        return any(m in key for m in markers)

    def _ask_agent(self, agent_id: str, question: str) -> str:
        question_clean = str(question or "").strip()
        if not question_clean:
            return "Question is empty. Usage: /ask agent_a What is your current plan?"

        ctx = self._build_agent_context(agent_id)
        fallback = {
            "answer": self._fallback_answer(agent_id, question_clean, ctx),
            "action_items": [],
            "risk_flags": [],
        }
        system_prompt = (
            "You are a trader agent in a systematic trading lab. "
            "Respond as a professional human trader. "
            "Use first-person voice, concrete and concise, no hype. "
            "Return JSON only with keys: answer, action_items, risk_flags."
        )
        user_prompt = (
            f"agent_id={agent_id}\n"
            f"question={question_clean}\n"
            f"context={json.dumps(ctx, ensure_ascii=False)}\n"
            "Keep answer grounded in current data and strategy state."
        )
        resp = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=0.3,
        )
        mode = str(resp.get("mode", "fallback") or "fallback")
        reason = str(resp.get("reason", "") or "")
        result = resp.get("result", fallback)
        if not isinstance(result, dict):
            result = fallback
        answer = str(result.get("answer", fallback["answer"]))
        action_items = result.get("action_items", [])
        risk_flags = result.get("risk_flags", [])
        if not isinstance(action_items, list):
            action_items = []
        if not isinstance(risk_flags, list):
            risk_flags = []

        self.identity.append_memory(
            agent_id=agent_id,
            memory_type="chat_user",
            content={"question": question_clean},
            ts=_now_iso(),
        )
        self.identity.append_memory(
            agent_id=agent_id,
            memory_type="chat_agent_reply",
            content={
                "mode": mode,
                "reason": reason,
                "answer": answer,
                "action_items": action_items,
                "risk_flags": risk_flags,
            },
            ts=_now_iso(),
        )

        if mode != "live" and self._is_token_budget_issue(reason):
            self._send_message(
                "[AgentLab] alert\n"
                f"agent={agent_id}\n"
                "OpenAI token/quota limit issue detected in chat response.\n"
                f"reason={reason}"
            )

        lines = [
            f"[{agent_id}]",
            answer,
        ]
        if action_items:
            lines.append("action_items:")
            for item in action_items[:5]:
                lines.append(f"- {str(item)}")
        if risk_flags:
            lines.append("risk_flags:")
            for item in risk_flags[:5]:
                lines.append(f"- {str(item)}")
        if mode != "live":
            lines.append(f"(mode={mode}, reason={reason or 'fallback'})")
        return "\n".join(lines)

    def _format_agent_status(self, agent_id: str) -> str:
        ctx = self._build_agent_context(agent_id)
        latest = ctx.get("latest_proposal") or {}
        orders = latest.get("orders") or []
        order_summary = "none"
        if orders:
            chunks = []
            for row in orders[:3]:
                chunks.append(
                    f"{row.get('side', '')} {row.get('symbol', '')} x{float(row.get('quantity', 0.0) or 0.0):.0f}"
                )
            order_summary = ", ".join(chunks)
        eq = ctx.get("latest_equity") or {}
        pos = ctx.get("positions") or []
        active = ctx.get("active_strategy") or {}
        version_tag = str(active.get("version_tag", "N/A"))
        score = float((latest or {}).get("strategy_version_id", 0) or 0)

        return (
            f"[{agent_id}] status\n"
            f"- strategy: {version_tag}\n"
            f"- latest_proposal: {latest.get('status', 'N/A')} ({latest.get('market', 'N/A')} {latest.get('session_date', 'N/A')})\n"
            f"- latest_orders: {order_summary}\n"
            f"- positions: {len(pos)}\n"
            f"- equity_krw: {float(eq.get('equity_krw', 0.0) or 0.0):.0f}\n"
            f"- drawdown: {float(eq.get('drawdown', 0.0) or 0.0):.4f}\n"
            f"- proposal_version_id: {score:.0f}"
        )

    def _format_all_status(self) -> str:
        agents = self.storage.list_agents()
        if not agents:
            return "No agents found. Run Agent Lab init first."
        lines = ["[AgentLab] all agents status"]
        for row in agents:
            aid = str(row.get("agent_id", ""))
            latest = self._latest_proposal(aid) or {}
            eq = self._latest_equity(aid) or {}
            lines.append(
                f"- {aid}: proposal={latest.get('status', 'N/A')} "
                f"market={latest.get('market', 'N/A')} date={latest.get('session_date', 'N/A')} "
                f"equity={float(eq.get('equity_krw', 0.0) or 0.0):.0f}"
            )
        council = self._latest_weekly_council()
        if council:
            lines.append(
                f"- latest_council: {council.get('week_id', 'N/A')} champion={council.get('champion_agent_id', 'N/A')}"
            )
        return "\n".join(lines)

    def _format_recent_memory(self, agent_id: str) -> str:
        rows = self.identity.load_recent_memories(agent_id, limit=10)
        if not rows:
            return f"[{agent_id}] no memory entries."
        lines = [f"[{agent_id}] recent memory"]
        for row in rows[-10:]:
            mem_type = str(row.get("memory_type", row.get("content", {}).get("memory_type", "unknown")))
            created_at = str(row.get("created_at", ""))
            content = row.get("content", {})
            if not isinstance(content, dict):
                content = {}
            compact = json.dumps(content, ensure_ascii=False)
            if len(compact) > 140:
                compact = compact[:140] + "..."
            lines.append(f"- {created_at} | {mem_type} | {compact}")
        return "\n".join(lines)

    def _format_directives(self, *, agent_id: Optional[str], status: Optional[str]) -> str:
        rows = self._list_directives(agent_id=agent_id, status=status, limit=12)
        title = "[AgentLab] directives"
        if agent_id:
            title += f" agent={agent_id}"
        if status:
            title += f" status={status.upper()}"
        if not rows:
            return title + "\n- (none)"
        lines = [title]
        for row in rows:
            did = int(row.get("directive_id", 0) or 0)
            aid = str(row.get("agent_id", ""))
            dtype = str(row.get("directive_type", ""))
            st = str(row.get("status", ""))
            requested = str(row.get("requested_at", row.get("created_at", "")))
            if dtype == "param_update":
                detail = f"{row.get('param_key', '')}={row.get('param_value_raw', '')}"
            else:
                detail = str(row.get("request_text", ""))
            if len(detail) > 72:
                detail = detail[:72] + "..."
            lines.append(
                f"- #{did} {st} {aid} [{dtype}] {detail} (at={requested})"
            )
        return "\n".join(lines)

    def _help_text(self) -> str:
        return (
            "[AgentLab Bot]\n"
            "Commands:\n"
            "/help\n"
            "/agents\n"
            "/status\n"
            "/status <agent_id>\n"
            "/plan <agent_id>\n"
            "/ask <agent_id> <question>\n"
            "/memory <agent_id>\n\n"
            "/directive <agent_id> <request>\n"
            "/setparam <agent_id> <param_key> <value> [note]\n"
            "/directives [agent_id] [pending|applied|rejected]\n"
            "/approve <directive_id> [note]\n"
            "/reject <directive_id> [reason]\n\n"
            "Examples:\n"
            "/status agent_a\n"
            "/plan agent_b\n"
            "/ask agent_c Why did you avoid the top-ranked symbol today?\n"
            "/directive agent_a Consider including sector leadership in your weekly review.\n"
            "/setparam agent_b min_strength 115 tighten entry quality\n"
            "/approve 42 looks good"
        )

    def _agent_list_text(self) -> str:
        rows = self.storage.list_agents()
        if not rows:
            return "No agents found. Run init first."
        lines = ["[AgentLab] agents"]
        for row in rows:
            lines.append(f"- {row.get('agent_id')}: {row.get('name')} ({row.get('role')})")
        return "\n".join(lines)

    def _authorized_chat(self, message: Dict[str, Any]) -> bool:
        chat = message.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if not self.chat_id:
            return False
        if chat_id != self.chat_id:
            return False
        configured_thread = str(self.thread_id or "").strip()
        if configured_thread:
            msg_thread = str(message.get("message_thread_id", "")).strip()
            if msg_thread and msg_thread != configured_thread:
                return False
        return True

    def _handle_message(self, message: Dict[str, Any]) -> None:
        if not self._authorized_chat(message):
            return
        text = str(message.get("text", "") or "").strip()
        if not text:
            return

        actor = self._actor_from_message(message)
        cmd, args = self._parse_command(text)
        out = ""
        if cmd in {"/start", "/help"}:
            out = self._help_text()
        elif cmd == "/agents":
            out = self._agent_list_text()
        elif cmd == "/status":
            if not args:
                out = self._format_all_status()
            else:
                aid = self._normalize_agent_id(args[0])
                if not aid:
                    out = "Unknown agent. Use /agents."
                else:
                    out = self._format_agent_status(aid)
        elif cmd == "/plan":
            if len(args) < 1:
                out = "Usage: /plan <agent_id>"
            else:
                aid = self._normalize_agent_id(args[0])
                if not aid:
                    out = "Unknown agent. Use /agents."
                else:
                    out = self._ask_agent(aid, "What is your current plan and rationale right now?")
        elif cmd == "/ask":
            if len(args) < 2:
                out = "Usage: /ask <agent_id> <question>"
            else:
                aid = self._normalize_agent_id(args[0])
                if not aid:
                    out = "Unknown agent. Use /agents."
                else:
                    question = " ".join(args[1:]).strip()
                    out = self._ask_agent(aid, question)
        elif cmd == "/memory":
            if len(args) < 1:
                out = "Usage: /memory <agent_id>"
            else:
                aid = self._normalize_agent_id(args[0])
                if not aid:
                    out = "Unknown agent. Use /agents."
                else:
                    out = self._format_recent_memory(aid)
        elif cmd == "/directive":
            if len(args) < 2:
                out = "Usage: /directive <agent_id> <request>"
            else:
                aid = self._normalize_agent_id(args[0])
                if not aid:
                    out = "Unknown agent. Use /agents."
                else:
                    request_text = " ".join(args[1:]).strip()
                    if not request_text:
                        out = "Request text is empty. Usage: /directive <agent_id> <request>"
                    else:
                        did = self._create_freeform_directive(aid, request_text, actor)
                        out = (
                            f"directive #{did} created (PENDING)\n"
                            f"agent={aid}\n"
                            f"request={request_text}\n"
                            "next: /approve <id> [note] or /reject <id> [reason]"
                        )
        elif cmd == "/setparam":
            if len(args) < 3:
                out = "Usage: /setparam <agent_id> <param_key> <value> [note]"
            else:
                aid = self._normalize_agent_id(args[0])
                if not aid:
                    out = "Unknown agent. Use /agents."
                else:
                    key = str(args[1]).strip()
                    value = str(args[2]).strip()
                    note = " ".join(args[3:]).strip() if len(args) > 3 else ""
                    ok, msg = self._create_param_directive(
                        agent_id=aid,
                        param_key=key,
                        param_value_raw=value,
                        requested_by=actor,
                        note=note,
                    )
                    out = msg
                    if ok:
                        out += "\nnext: /approve <id> [note] or /reject <id> [reason]"
        elif cmd == "/directives":
            aid = None
            status = None
            if len(args) >= 1:
                maybe_agent = self._normalize_agent_id(args[0])
                if maybe_agent:
                    aid = maybe_agent
                else:
                    status = str(args[0]).strip()
            if len(args) >= 2:
                status = str(args[1]).strip()
            out = self._format_directives(agent_id=aid, status=status)
        elif cmd in {"/approve", "/apply"}:
            if len(args) < 1:
                out = "Usage: /approve <directive_id> [note]"
            else:
                try:
                    did = int(args[0])
                except Exception:
                    did = 0
                if did <= 0:
                    out = "directive_id must be a positive integer."
                else:
                    directive = self._directive_by_id(did)
                    if directive is None:
                        out = f"directive #{did} not found."
                    else:
                        note = " ".join(args[1:]).strip()
                        ok, msg = self._apply_directive(directive, actor, note)
                        out = msg
        elif cmd == "/reject":
            if len(args) < 1:
                out = "Usage: /reject <directive_id> [reason]"
            else:
                try:
                    did = int(args[0])
                except Exception:
                    did = 0
                if did <= 0:
                    out = "directive_id must be a positive integer."
                else:
                    directive = self._directive_by_id(did)
                    if directive is None:
                        out = f"directive #{did} not found."
                    else:
                        note = " ".join(args[1:]).strip()
                        ok, msg = self._reject_directive(directive, actor, note)
                        out = msg
        elif cmd:
            out = "Unknown command. Use /help."
        else:
            if ":" in text:
                head, tail = text.split(":", 1)
                aid = self._normalize_agent_id(head.strip())
                if aid and tail.strip():
                    out = self._ask_agent(aid, tail.strip())
                else:
                    out = "Use /help for supported commands."
            else:
                out = "Use /help for supported commands."

        thread_id = message.get("message_thread_id")
        self._send_message(out, thread_id=str(thread_id) if thread_id is not None else None)

    def _poll_once(self) -> int:
        if not self.telegram_enabled:
            raise RuntimeError("Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        params = {
            "offset": self.offset,
            "timeout": self.poll_timeout,
            "allowed_updates": json.dumps(["message"]),
        }
        data = self._api_get("getUpdates", params=params)
        updates = data.get("result") or []
        handled = 0
        for upd in updates:
            try:
                update_id = int(upd.get("update_id", 0))
            except Exception:
                update_id = 0
            if update_id > 0:
                self._save_offset(update_id + 1)
            msg = upd.get("message")
            if isinstance(msg, dict):
                self._handle_message(msg)
                handled += 1
                self.processed_updates += 1
        return handled

    def run(self, *, once: bool = False) -> Dict[str, Any]:
        started_at = _now_iso()
        self.storage.log_event(
            event_type="telegram_chat_worker_start",
            payload={"started_at": started_at, "offset": self.offset},
            created_at=started_at,
        )
        if once:
            handled = self._poll_once()
            finished = _now_iso()
            payload = {
                "mode": "once",
                "handled_updates": handled,
                "processed_updates_total": self.processed_updates,
                "started_at": started_at,
                "finished_at": finished,
            }
            self.storage.log_event("telegram_chat_worker_once", payload, finished)
            return payload

        while True:
            try:
                self._poll_once()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                now = _now_iso()
                self.storage.log_event(
                    event_type="telegram_chat_worker_error",
                    payload={"error": self._sanitize_error(exc)},
                    created_at=now,
                )
                time.sleep(5.0)
                continue
            time.sleep(self.idle_sleep)

        finished = _now_iso()
        payload = {
            "mode": "loop",
            "processed_updates_total": self.processed_updates,
            "started_at": started_at,
            "finished_at": finished,
        }
        self.storage.log_event("telegram_chat_worker_stop", payload, finished)
        return payload


def run_telegram_chat_worker(
    *,
    project_root: str | Path,
    poll_timeout: int = 25,
    idle_sleep: float = 1.0,
    memory_limit: int = 20,
    once: bool = False,
) -> Dict[str, Any]:
    runtime = TelegramChatRuntime(
        project_root=project_root,
        poll_timeout=poll_timeout,
        idle_sleep=idle_sleep,
        memory_limit=memory_limit,
    )
    try:
        return runtime.run(once=once)
    finally:
        runtime.close()
