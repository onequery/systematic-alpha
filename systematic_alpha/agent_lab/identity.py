from __future__ import annotations

import json
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

    def ensure_identity(self, profile: AgentProfile) -> Path:
        path = self.identity_path(profile.agent_id)
        if path.exists():
            return path
        body = [
            f"# {profile.name} ({profile.agent_id})",
            "",
            f"- Role: {profile.role}",
            f"- Risk Style: {profile.risk_style}",
            f"- Allocated Capital (KRW): {profile.allocated_capital_krw:.2f}",
            "",
            "## Philosophy",
            profile.philosophy.strip(),
            "",
            "## Hard Constraints",
            "```json",
            json.dumps(profile.constraints, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Banned Behaviors",
            "- No live trading outside explicit approval flow.",
            "- No leverage or shorting unless strategy registry explicitly allows it.",
            "- No out-of-scope parameter edits beyond allowed list.",
            "",
            "## Running Lessons (last 4 weeks)",
            "- (to be appended by weekly council)",
            "",
        ]
        path.write_text("\n".join(body), encoding="utf-8")
        return path

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
            return records[-limit:]
        db_rows = self.storage.list_agent_memories(agent_id, limit=limit)
        return list(reversed(db_rows))

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
        return {
            "agent_id": agent_id,
            "identity_markdown": identity,
            "recent_memories": memories,
            "latest_checkpoint": checkpoint,
        }
