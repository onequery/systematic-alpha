from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from systematic_alpha.agent_lab.llm_client import LLMClient
from systematic_alpha.agent_lab.storage import AgentLabStorage


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class HealIssue:
    code: str
    severity: str
    summary: str
    evidence: List[str]
    auto_patchable: bool


class DataReceptionSelfHealer:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.state_root = self.project_root / "state" / "agent_lab"
        self.out_root = self.project_root / "out" / "agent_lab"
        self.storage = AgentLabStorage(self.state_root / "agent_lab.sqlite")
        self.llm = LLMClient(self.storage)

    def close(self) -> None:
        self.storage.close()

    def _read_text(self, path: Optional[Path]) -> str:
        if path is None or not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _read_json(self, path: Optional[Path]) -> Dict[str, Any]:
        if path is None or not path.exists():
            return {}
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {}

    def _detect_issues(self, combined_log_text: str, output_payload: Dict[str, Any]) -> List[HealIssue]:
        issues: List[HealIssue] = []
        lines = combined_log_text.splitlines()
        evidence = "\n".join(lines[-120:])

        if (
            "ProxyError" in combined_log_text
            and ("127.0.0.1', port=9" in combined_log_text or "127.0.0.1:9" in combined_log_text)
        ):
            issues.append(
                HealIssue(
                    code="PROXY_LOOPBACK_BLOCK",
                    severity="critical",
                    summary="Network calls are blocked by a broken local proxy (127.0.0.1:9).",
                    evidence=[evidence[-1500:]],
                    auto_patchable=True,
                )
            )

        if re.search(r"KeyError\('(?:NASD|NASDAQ|NYSE|AMEX|NYS|AMS)'\)", combined_log_text):
            issues.append(
                HealIssue(
                    code="US_EXCHANGE_KEYERROR",
                    severity="critical",
                    summary="US exchange mapping key error detected in mojito integration path.",
                    evidence=[evidence[-1500:]],
                    auto_patchable=True,
                )
            )

        invalid_reason = str(output_payload.get("invalid_reason", "") or "")
        rq = output_payload.get("realtime_quality") or {}
        coverage = float(rq.get("coverage_ratio", 0.0) or 0.0)
        eligible = int(rq.get("eligible_count", 0) or 0)
        total = int(rq.get("total_count", 0) or 0)
        if (
            invalid_reason.startswith("realtime_coverage_too_low")
            and coverage <= 1e-6
            and eligible == 0
            and total > 0
        ):
            issues.append(
                HealIssue(
                    code="REALTIME_STREAM_EMPTY",
                    severity="high",
                    summary=(
                        "Realtime stream quality is empty while stage1 had symbols. "
                        "May be market closed/holiday or feed parser issue."
                    ),
                    evidence=[
                        f"invalid_reason={invalid_reason}",
                        f"coverage_ratio={coverage}",
                        f"eligible={eligible}/{total}",
                    ],
                    auto_patchable=False,
                )
            )

        return issues

    def _contains_marker(self, rel_path: str, marker: str) -> bool:
        path = self.project_root / rel_path
        if not path.exists():
            return False
        return marker in self._read_text(path)

    def _write_text(self, rel_path: str, content: str) -> None:
        path = self.project_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _inject_once(self, src: str, needle: str, payload: str) -> str:
        if payload in src:
            return src
        return src.replace(needle, needle + payload, 1)

    def _attempt_patch_proxy_guard(self) -> Dict[str, Any]:
        marker = "self-heal:network-guard-v1"
        changed: List[str] = []
        required = ["systematic_alpha/network_env.py", "main.py", "systematic_alpha/agent_lab/cli.py"]
        missing = [rel for rel in required if not self._contains_marker(rel, marker)]
        if not missing:
            return {"applied": False, "status": "already_present", "changed_files": changed}

        if "systematic_alpha/network_env.py" in missing:
            network_template = """from __future__ import annotations

import os
from typing import Dict


PROXY_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]


def apply_network_env_guard() -> Dict[str, str]:
    # self-heal:network-guard-v1
    mode = str(os.getenv("SYSTEMATIC_ALPHA_PROXY_MODE", "auto")).strip().lower()
    if mode in {"off", "disabled", "keep"}:
        return {}
    removed: Dict[str, str] = {}
    for key in PROXY_KEYS:
        value = os.getenv(key)
        if value is None:
            continue
        clean = str(value).strip().lower()
        should_remove = False
        if mode in {"clear", "clear_all", "force"}:
            should_remove = True
        elif mode == "auto":
            should_remove = "127.0.0.1:9" in clean or "localhost:9" in clean
        if should_remove:
            removed[key] = value
            os.environ.pop(key, None)
    return removed
"""
            self._write_text("systematic_alpha/network_env.py", network_template)
            changed.append("systematic_alpha/network_env.py")

        if "main.py" in missing:
            main_path = self.project_root / "main.py"
            src = self._read_text(main_path)
            if "from systematic_alpha.network_env import apply_network_env_guard" not in src:
                src = self._inject_once(
                    src,
                    "from systematic_alpha.models import StrategyConfig\n",
                    "from systematic_alpha.network_env import apply_network_env_guard\n",
                )
            if "apply_network_env_guard()" not in src:
                src = src.replace(
                    "def main() -> None:\n    load_dotenv(\".env\", override=False)\n",
                    "def main() -> None:\n    load_dotenv(\".env\", override=False)\n"
                    "    # self-heal:network-guard-v1\n"
                    "    apply_network_env_guard()\n",
                    1,
                )
            self._write_text("main.py", src)
            changed.append("main.py")

        if "systematic_alpha/agent_lab/cli.py" in missing:
            cli_path = self.project_root / "systematic_alpha" / "agent_lab" / "cli.py"
            src = self._read_text(cli_path)
            if "from systematic_alpha.network_env import apply_network_env_guard" not in src:
                src = self._inject_once(
                    src,
                    "from systematic_alpha.agent_lab.telegram_chat import run_telegram_chat_worker\n",
                    "from systematic_alpha.network_env import apply_network_env_guard\n",
                )
            if "apply_network_env_guard()" not in src:
                src = src.replace(
                    "def main() -> None:\n    load_dotenv(\".env\", override=False)\n",
                    "def main() -> None:\n    load_dotenv(\".env\", override=False)\n"
                    "    # self-heal:network-guard-v1\n"
                    "    apply_network_env_guard()\n",
                    1,
                )
            self._write_text("systematic_alpha/agent_lab/cli.py", src)
            changed.append("systematic_alpha/agent_lab/cli.py")

        return {
            "applied": len(changed) > 0,
            "status": "patched" if changed else "manual_required",
            "changed_files": changed,
            "reason": "" if changed else "failed to patch network-guard markers",
        }

    def _attempt_patch_us_exchange_resolver(self) -> Dict[str, Any]:
        marker = "self-heal:us-exchange-resolver-v1"
        changed: List[str] = []
        required = ["systematic_alpha/selector_us.py", "systematic_alpha/agent_lab/paper_broker.py"]
        missing = [rel for rel in required if not self._contains_marker(rel, marker)]
        if not missing:
            return {"applied": False, "status": "already_present", "changed_files": changed}

        # Safe mode for exchange resolver: avoid free-form rewriting.
        return {
            "applied": False,
            "status": "manual_required",
            "changed_files": changed,
            "reason": "us-exchange resolver marker missing in core files",
        }

    def _validate_changed_files(self, changed_files: List[str]) -> Dict[str, Any]:
        if not changed_files:
            return {"ok": True, "command": "", "output": ""}
        abs_paths = [str((self.project_root / rel).resolve()) for rel in changed_files]
        cmd = [sys.executable, "-m", "compileall", *abs_paths]
        proc = subprocess.run(
            cmd,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "command": " ".join(cmd),
            "output": (proc.stdout or "") + (proc.stderr or ""),
            "returncode": proc.returncode,
        }

    def _llm_diagnosis(self, issues: List[HealIssue], market: str, run_date: str) -> Dict[str, Any]:
        fallback = {
            "summary": "LLM unavailable. Rule-based self-heal diagnosis only.",
            "next_actions": ["Inspect run log and output json manually if issue persists."],
        }
        if not issues:
            return fallback
        payload = [
            {
                "code": i.code,
                "severity": i.severity,
                "summary": i.summary,
                "evidence": i.evidence[:3],
            }
            for i in issues
        ]
        resp = self.llm.generate_json(
            system_prompt=(
                "You are a production trading infra SRE assistant. "
                "Return JSON only with keys: summary, next_actions."
            ),
            user_prompt=(
                f"market={market}\n"
                f"run_date={run_date}\n"
                f"issues={json.dumps(payload, ensure_ascii=False)}\n"
                "Focus on safe, immediate mitigation and validation."
            ),
            fallback=fallback,
            temperature=0.1,
        )
        result = resp.get("result", fallback)
        if not isinstance(result, dict):
            result = fallback
        result["mode"] = resp.get("mode", "fallback")
        result["reason"] = resp.get("reason", "")
        return result

    def run(
        self,
        *,
        market: str,
        run_date: str,
        log_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        failure_tail: str = "",
        auto_apply: bool = True,
    ) -> Dict[str, Any]:
        market_upper = str(market or "").strip().upper()
        log_file = Path(log_path).resolve() if log_path else None
        output_file = Path(output_json_path).resolve() if output_json_path else None

        log_text = self._read_text(log_file)
        if failure_tail:
            log_text = f"{log_text}\n{failure_tail}"
        output_payload = self._read_json(output_file)
        issues = self._detect_issues(log_text, output_payload)

        auto_actions: List[Dict[str, Any]] = []
        changed_files: List[str] = []
        for issue in issues:
            if not auto_apply or not issue.auto_patchable:
                continue
            if issue.code == "PROXY_LOOPBACK_BLOCK":
                action = self._attempt_patch_proxy_guard()
            elif issue.code == "US_EXCHANGE_KEYERROR":
                action = self._attempt_patch_us_exchange_resolver()
            else:
                action = {"applied": False, "status": "unsupported", "changed_files": []}
            auto_actions.append({"issue_code": issue.code, **action})
            changed_files.extend(action.get("changed_files", []))

        validate = self._validate_changed_files(changed_files)
        llm_diag = self._llm_diagnosis(issues, market_upper, run_date)
        applied_count = sum(1 for a in auto_actions if bool(a.get("applied")))

        result = {
            "market": market_upper,
            "run_date": run_date,
            "issue_count": len(issues),
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "summary": i.summary,
                    "evidence": i.evidence[:3],
                    "auto_patchable": i.auto_patchable,
                }
                for i in issues
            ],
            "auto_apply": auto_apply,
            "applied_count": applied_count,
            "auto_actions": auto_actions,
            "compile_validation": validate,
            "diagnosis": llm_diag,
            "log_path": str(log_file) if log_file else "",
            "output_json_path": str(output_file) if output_file else "",
            "generated_at": _now_iso(),
        }

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.out_root / run_date
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = out_dir / f"self_heal_{market_upper.lower()}_{stamp}.json"
        artifact.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["artifact_path"] = str(artifact)

        self.storage.log_event(
            event_type="self_heal_run",
            payload={
                "market": market_upper,
                "run_date": run_date,
                "issue_count": len(issues),
                "applied_count": applied_count,
                "artifact_path": str(artifact),
            },
            created_at=_now_iso(),
        )
        return result


def run_data_reception_self_heal(
    *,
    project_root: str | Path,
    market: str,
    run_date: str,
    log_path: Optional[str] = None,
    output_json_path: Optional[str] = None,
    failure_tail: str = "",
    auto_apply: bool = True,
) -> Dict[str, Any]:
    healer = DataReceptionSelfHealer(project_root)
    try:
        return healer.run(
            market=market,
            run_date=run_date,
            log_path=log_path,
            output_json_path=output_json_path,
            failure_tail=failure_tail,
            auto_apply=auto_apply,
        )
    finally:
        healer.close()
