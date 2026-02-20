from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from systematic_alpha.agent_lab.schemas import AgentProfile, now_iso
from systematic_alpha.agent_lab.storage import AgentLabStorage


class AgentIdentityStore:
    def __init__(self, base_dir: str | Path, storage: AgentLabStorage):
        self.base_dir = Path(base_dir)
        self.storage = storage
        self.agents_dir = self.base_dir / "agents"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _agent_dir(self, agent_id: str) -> Path:
        path = self.agents_dir / agent_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def identity_path(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "identity.md"

    def memory_path(self, agent_id: str) -> Path:
        return self._agent_dir(agent_id) / "memory.jsonl"

    @staticmethod
    def _truthy(value: str) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _legacy_identity_markers() -> List[str]:
        return [
            "## Hard Constraints",
            "max_daily_picks",
            "allowed_params",
            "No live trading outside explicit approval flow.",
            "No leverage or shorting unless strategy registry explicitly allows it.",
        ]

    @classmethod
    def _needs_identity_refresh(cls, text: str) -> bool:
        low = str(text or "").lower()
        return any(marker.lower() in low for marker in cls._legacy_identity_markers())

    def _render_identity(self, profile: AgentProfile) -> str:
        body = [
            f"# {profile.name} ({profile.agent_id})",
            "",
            f"- Role: {profile.role}",
            f"- Risk Style: {profile.risk_style}",
            f"- Allocated Capital (KRW): {profile.allocated_capital_krw:.2f}",
            "- Budget Isolation: Enabled (this agent can only use its own ledger/cash/equity).",
            "- Cross-Market Mode: KR/US simultaneous operation with allocation-aware sizing.",
            "- Autonomy Mode: Max-freedom realtime (adaptive intraday monitoring + auto execution).",
            "",
            "## Philosophy",
            profile.philosophy.strip(),
            "",
            "## Operating Rules",
            "- Never access or consume another agent's budget/cash/positions.",
            "- Never modify another agent's identity/memory/state.",
            "- Keep all execution and discussion events logged for reproducibility.",
            "",
            "## Running Lessons (last 4 weeks)",
            "- (to be appended by weekly council)",
            "",
        ]
        return "\n".join(body)

    def ensure_identity(self, profile: AgentProfile, force_refresh: bool = False) -> Path:
        path = self.identity_path(profile.agent_id)
        requested = force_refresh or self._truthy(os.getenv("AGENT_LAB_FORCE_REFRESH_IDENTITY", "1"))
        if path.exists():
            if not requested:
                try:
                    existing = path.read_text(encoding="utf-8")
                except Exception:
                    existing = ""
                if not self._needs_identity_refresh(existing):
                    return path
        rendered = self._render_identity(profile)
        path.write_text(rendered, encoding="utf-8")
        return path

    @staticmethod
    def _is_legacy_memory(record: Dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            return True
        mem_type = str(record.get("memory_type", "")).strip().lower()
        content = record.get("content", {})
        status = ""
        if isinstance(content, dict):
            status = str(content.get("status", "")).strip().upper()
        if mem_type == "proposal" and status == "PENDING_APPROVAL":
            return True
        blob = json.dumps(record, ensure_ascii=False).lower()
        legacy_markers = [
            "pending_approval",
            "max_daily_picks=3",
            "exposure_cap_ratio=0.95",
            "collect_seconds=600",
            "scheduled_daily",
            "not always-on loop",
            "hard constraints",
            "approval flow",
        ]
        return any(marker in blob for marker in legacy_markers)

    def sanitize_memory_file(self, agent_id: str) -> Dict[str, int]:
        path = self.memory_path(agent_id)
        if not path.exists():
            return {"before": 0, "after": 0, "removed": 0}
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        before = len(rows)
        filtered = [row for row in rows if not self._is_legacy_memory(row)]
        after = len(filtered)
        with path.open("w", encoding="utf-8") as f:
            for row in filtered:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"before": before, "after": after, "removed": max(0, before - after)}

    def append_memory(self, agent_id: str, memory_type: str, content: Dict[str, Any], ts: Optional[str] = None) -> None:
        created_at = ts or now_iso()
        mem = {"created_at": created_at, "memory_type": memory_type, "content": content}
        mem_path = self.memory_path(agent_id)
        with mem_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(mem, ensure_ascii=False) + "\n")
        self.storage.insert_agent_memory(
            agent_id=agent_id,
            memory_type=memory_type,
            content=content,
            created_at=created_at,
        )

    def load_recent_memories(self, agent_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        path = self.memory_path(agent_id)
        records: List[Dict[str, Any]] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
        if records:
            records = [row for row in records if not self._is_legacy_memory(row)]
            return records[-limit:]
        db_rows = self.storage.list_agent_memories(agent_id, limit=limit)
        sanitized = [row for row in list(reversed(db_rows)) if not self._is_legacy_memory(row)]
        return sanitized

    def save_checkpoint(self, week_id: str, payload: Dict[str, Any]) -> Path:
        path = self.checkpoints_dir / f"checkpoint_{week_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        files = sorted(self.checkpoints_dir.glob("checkpoint_*.json"))
        if not files:
            return None
        latest = files[-1]
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None

    def build_warm_start_context(self, agent_id: str, memory_limit: int = 20) -> Dict[str, Any]:
        identity = ""
        ipath = self.identity_path(agent_id)
        if ipath.exists():
            identity = ipath.read_text(encoding="utf-8")
        memories = self.load_recent_memories(agent_id, limit=memory_limit)
        checkpoint = self.load_latest_checkpoint()
        checkpoint_summary = None
        if isinstance(checkpoint, dict):
            moderator = checkpoint.get("discussion", {}).get("moderator", {}) if isinstance(checkpoint.get("discussion"), dict) else {}
            checkpoint_summary = {
                "week_id": checkpoint.get("week_id"),
                "champion_agent_id": checkpoint.get("champion_agent_id"),
                "promoted_versions": checkpoint.get("promoted_versions"),
                "score_board": checkpoint.get("score_board"),
                "moderator_summary": moderator.get("summary") if isinstance(moderator, dict) else "",
            }
        return {
            "agent_id": agent_id,
            "identity_markdown": identity,
            "recent_memories": memories,
            "latest_checkpoint": checkpoint_summary,
        }
