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

    @staticmethod
    def _clip_text(text: Any, limit: int = 220) -> str:
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        return value[:limit] + "..."

    @staticmethod
    def _replace_running_lessons_section(identity_text: str, lessons: List[str]) -> str:
        lines = str(identity_text or "").splitlines()
        marker = "## Running Lessons (last 4 weeks)"
        start = -1
        for idx, line in enumerate(lines):
            if line.strip() == marker:
                start = idx
                break
        if start < 0:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(marker)
            lines.append("- (to be appended by weekly council)")
            start = len(lines) - 2

        end = len(lines)
        for idx in range(start + 1, len(lines)):
            if lines[idx].startswith("## "):
                end = idx
                break

        block = [marker]
        if lessons:
            block.extend(lessons)
        else:
            block.append("- (to be appended by weekly council)")

        rebuilt = lines[:start] + block + lines[end:]
        return "\n".join(rebuilt).rstrip() + "\n"

    def _build_weekly_lesson(self, agent_id: str, week_id: str, decision: Dict[str, Any]) -> str:
        rows_scored = list(decision.get("rows_scored") or [])
        row = next((r for r in rows_scored if str(r.get("agent_id", "")) == agent_id), {}) if rows_scored else {}
        score = float(row.get("score", 0.0) or 0.0)
        violations = int(row.get("risk_violations", 0) or 0)
        proposals = int(row.get("proposal_count", 0) or 0)

        discussion = decision.get("discussion") if isinstance(decision.get("discussion"), dict) else {}
        moderator = discussion.get("moderator") if isinstance(discussion.get("moderator"), dict) else {}
        consensus_actions = moderator.get("consensus_actions") if isinstance(moderator.get("consensus_actions"), list) else []
        risk_watch = moderator.get("risk_watch") if isinstance(moderator.get("risk_watch"), list) else []
        action = self._clip_text(consensus_actions[0], limit=90) if consensus_actions else "리스크 이벤트와 회전율을 우선 점검"
        watch = self._clip_text(risk_watch[0], limit=90) if risk_watch else "위험 신호 발생 시 즉시 포지션/노출을 점검"

        suggestion_map = discussion.get("agent_param_suggestions") if isinstance(discussion.get("agent_param_suggestions"), dict) else {}
        params = suggestion_map.get(agent_id) if isinstance(suggestion_map.get(agent_id), dict) else {}
        pos_cap = params.get("position_cap_ratio")
        exp_cap = params.get("exposure_cap_ratio")

        cap_note_parts: List[str] = []
        try:
            if pos_cap is not None:
                cap_note_parts.append(f"포지션한도={float(pos_cap):.2f}")
        except Exception:
            pass
        try:
            if exp_cap is not None:
                cap_note_parts.append(f"노출한도={float(exp_cap):.2f}")
        except Exception:
            pass
        cap_note = ", ".join(cap_note_parts) if cap_note_parts else "한도 파라미터 유지"

        return (
            f"- [{week_id}] 점수 {score:.3f}, 리스크위반 {violations}건, 제안 {proposals}건. "
            f"{cap_note}. 공통합의: {action}. 리스크주시: {watch}."
        )

    def update_running_lessons(self, week_id: str, decision: Dict[str, Any], max_weeks: int = 4) -> None:
        if not isinstance(decision, dict):
            return
        agents = self.storage.list_agents()
        for row in agents:
            agent_id = str(row.get("agent_id", "")).strip()
            if not agent_id:
                continue
            path = self.identity_path(agent_id)
            if not path.exists():
                continue
            try:
                identity_text = path.read_text(encoding="utf-8")
            except Exception:
                continue

            lesson_line = self._build_weekly_lesson(agent_id, week_id, decision)
            marker = "## Running Lessons (last 4 weeks)"
            lines = identity_text.splitlines()
            start = -1
            for idx, line in enumerate(lines):
                if line.strip() == marker:
                    start = idx
                    break
            existing_lessons: List[str] = []
            if start >= 0:
                for idx in range(start + 1, len(lines)):
                    line = lines[idx]
                    if line.startswith("## "):
                        break
                    stripped = line.strip()
                    if not stripped or stripped == "- (to be appended by weekly council)":
                        continue
                    if stripped.startswith("- ["):
                        existing_lessons.append(stripped)
            existing_lessons = [x for x in existing_lessons if not x.startswith(f"- [{week_id}]")]
            merged = [lesson_line] + existing_lessons
            merged = merged[: max(1, int(max_weeks))]
            updated = self._replace_running_lessons_section(identity_text, merged)
            path.write_text(updated, encoding="utf-8")

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
