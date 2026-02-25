import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from systematic_alpha.agent_lab.orchestrator import AgentLabOrchestrator


def _sample_signal_payload(market: str) -> dict:
    return {
        "market": market,
        "generated_at": "2026-02-18T09:10:00+09:00",
        "signal_valid": True,
        "invalid_reason": None,
        "final": [],
        "all_ranked": [
            {
                "rank": 1,
                "code": "005930" if market == "KR" else "AAPL",
                "name": "SAMSUNG" if market == "KR" else "APPLE",
                "score": 7,
                "max_score": 8,
                "recommendation_score": 87.5,
                "metrics": {"latest_price": 70000 if market == "KR" else 190.0},
            },
            {
                "rank": 2,
                "code": "000660" if market == "KR" else "MSFT",
                "name": "SKHYNIX" if market == "KR" else "MICROSOFT",
                "score": 6,
                "max_score": 8,
                "recommendation_score": 75.0,
                "metrics": {"latest_price": 120000 if market == "KR" else 410.0},
            },
            {
                "rank": 3,
                "code": "035420" if market == "KR" else "NVDA",
                "name": "NAVER" if market == "KR" else "NVIDIA",
                "score": 6,
                "max_score": 8,
                "recommendation_score": 73.0,
                "metrics": {"latest_price": 200000 if market == "KR" else 720.0},
            },
            {
                "rank": 4,
                "code": "051910" if market == "KR" else "AMZN",
                "name": "LGCHEM" if market == "KR" else "AMAZON",
                "score": 5,
                "max_score": 8,
                "recommendation_score": 62.5,
                "metrics": {"latest_price": 350000 if market == "KR" else 170.0},
            },
            {
                "rank": 5,
                "code": "005380" if market == "KR" else "META",
                "name": "HYUNDAI" if market == "KR" else "META",
                "score": 5,
                "max_score": 8,
                "recommendation_score": 61.5,
                "metrics": {"latest_price": 180000 if market == "KR" else 490.0},
            },
        ],
    }


def _write_signal_file(root: Path, market: str, run_date: str) -> None:
    market_lc = market.lower()
    result_dir = root / "out" / market_lc / run_date / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = _sample_signal_payload(market)
    out_json = result_dir / f"{market_lc}_daily_{run_date}_091001.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class AgentLabOrchestratorTests(unittest.TestCase):
    def test_init_ingest_propose_auto_execute_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_date = "20260218"
            _write_signal_file(root, "KR", run_date)

            orch = AgentLabOrchestrator(project_root=root)
            try:
                with patch.dict("os.environ", {"AGENT_LAB_ENFORCE_MARKET_HOURS": "0"}):
                    init_payload = orch.init_lab(capital_krw=10_000_000, agents=3)
                    self.assertEqual(3, len(init_payload["agents"]))

                    ingest_payload = orch.ingest_session(market="KR", yyyymmdd=run_date)
                    self.assertEqual("SIGNAL_OK", ingest_payload["status_code"])

                    proposal_payload = orch.propose_orders(market="KR", yyyymmdd=run_date)
                    self.assertEqual(3, len(proposal_payload["proposals"]))

                    statuses = {str(x.get("status", "")) for x in proposal_payload["proposals"]}
                    self.assertIn("EXECUTED", statuses)

                    review_payload = orch.daily_review(run_date)
                    self.assertEqual(run_date, review_payload["date"])
                    self.assertEqual(3, len(review_payload["rows"]))
            finally:
                orch.close()

    def test_cross_market_plan_caps_kr_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_date = "20260218"
            _write_signal_file(root, "KR", run_date)

            orch = AgentLabOrchestrator(project_root=root)
            try:
                with patch.dict("os.environ", {"AGENT_LAB_ENFORCE_MARKET_HOURS": "0"}):
                    orch.init_lab(capital_krw=10_000_000, agents=1)
                    orch.ingest_session(market="KR", yyyymmdd=run_date)
                    proposal_payload = orch.propose_orders(market="KR", yyyymmdd=run_date)
                    self.assertEqual(1, len(proposal_payload["proposals"]))

                    proposal = proposal_payload["proposals"][0]
                    plan = proposal.get("cross_market_plan", {})
                    self.assertIn("target_weights", plan)
                    self.assertIn("KR", plan["target_weights"])
                    self.assertIn("US", plan["target_weights"])

                    budget_cap = float(plan.get("buy_budget_cap_krw", 0.0) or 0.0)
                    self.assertGreater(budget_cap, 0.0)
                    self.assertLessEqual(budget_cap, 6_700_000.0)

                    buy_notional = 0.0
                    for order in list(proposal.get("orders") or []):
                        if str(order.get("side", "")).upper() != "BUY":
                            continue
                        buy_notional += float(order.get("quantity", 0.0) or 0.0) * float(order.get("reference_price", 0.0) or 0.0)
                    self.assertLessEqual(buy_notional, budget_cap + 1.0)
            finally:
                orch.close()

    def test_propose_orders_skips_outside_market_hours(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_date = "20260218"
            _write_signal_file(root, "KR", run_date)

            orch = AgentLabOrchestrator(project_root=root)
            try:
                orch.init_lab(capital_krw=10_000_000, agents=1)
                with patch.dict("os.environ", {"AGENT_LAB_ENFORCE_MARKET_HOURS": "1"}):
                    with patch.object(orch, "_is_market_open_now", return_value=(False, "test_closed_window")):
                        payload = orch.propose_orders(market="KR", yyyymmdd=run_date)
                self.assertTrue(bool(payload.get("skipped")))
                self.assertEqual("outside_market_hours", payload.get("reason"))
                self.assertEqual([], payload.get("proposals"))
            finally:
                orch.close()

    def test_weekly_council_live_result_is_not_overwritten_by_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            orch = AgentLabOrchestrator(project_root=root)
            try:
                week_id = "2026-W08"
                live_decision = {
                    "week_id": week_id,
                    "champion_agent_id": "agent_b",
                    "discussion": {
                        "rounds": [
                            {
                                "round": 1,
                                "phase": "opening",
                                "speeches": [
                                    {
                                        "agent_id": "agent_a",
                                        "mode": "live",
                                        "thesis": "live thesis",
                                    }
                                ],
                            }
                        ],
                        "moderator": {"mode": "live", "summary": "live summary"},
                    },
                    "llm_alerts": [],
                }
                fallback_decision = {
                    "week_id": week_id,
                    "champion_agent_id": "agent_a",
                    "discussion": {
                        "rounds": [
                            {
                                "round": 1,
                                "phase": "opening",
                                "speeches": [
                                    {
                                        "agent_id": "agent_a",
                                        "mode": "fallback",
                                        "thesis": "fallback thesis",
                                    }
                                ],
                            }
                        ],
                        "moderator": {"mode": "fallback", "summary": "fallback summary"},
                    },
                    "llm_alerts": [{"agent_id": "agent_a", "phase": "opening", "reason": "llm_disabled_or_unavailable"}],
                }

                orch.storage.upsert_weekly_council(
                    week_id=week_id,
                    champion_agent_id="agent_b",
                    decision=live_decision,
                    markdown="live markdown",
                    created_at="2026-02-20T04:16:45",
                )
                orch.storage.upsert_weekly_council(
                    week_id=week_id,
                    champion_agent_id="agent_a",
                    decision=fallback_decision,
                    markdown="fallback markdown",
                    created_at="2026-02-20T15:56:56",
                )

                row = orch.storage.query_one(
                    "SELECT champion_agent_id, decision_json, markdown, created_at FROM weekly_councils WHERE week_id = ?",
                    (week_id,),
                )
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual("agent_b", row["champion_agent_id"])
                self.assertEqual("live markdown", row["markdown"])
                self.assertEqual("2026-02-20T04:16:45", row["created_at"])
                stored = json.loads(str(row["decision_json"]))
                speech_mode = (
                    stored.get("discussion", {})
                    .get("rounds", [{}])[0]
                    .get("speeches", [{}])[0]
                    .get("mode")
                )
                self.assertEqual("live", speech_mode)
            finally:
                orch.close()

    def test_weekly_council_immediately_applies_params_to_active_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            orch = AgentLabOrchestrator(project_root=root)
            try:
                orch.init_lab(capital_krw=10_000_000, agents=3)

                expected_intervals = {
                    "agent_a": 75,
                    "agent_b": 90,
                    "agent_c": 120,
                }
                expected_picks = {
                    "agent_a": 11,
                    "agent_b": 9,
                    "agent_c": 7,
                }

                def _fake_live_weekly_debate(**kwargs):
                    profiles = list(kwargs.get("agent_profiles", []))
                    active_params_map = dict(kwargs.get("active_params_map", {}))
                    suggestions = {}
                    for profile in profiles:
                        aid = profile.agent_id
                        params = dict(active_params_map.get(aid, {}))
                        params["intraday_monitor_interval_sec"] = expected_intervals[aid]
                        params["max_daily_picks"] = expected_picks[aid]
                        suggestions[aid] = params
                    opening = []
                    rebuttal = []
                    for profile in profiles:
                        aid = profile.agent_id
                        opening.append(
                            {
                                "agent_id": aid,
                                "mode": "live",
                                "reason": "",
                                "thesis": f"{aid} opening",
                                "risk_notes": [f"{aid} risk note"],
                                "param_changes": {},
                                "confidence": 0.7,
                            }
                        )
                        rebuttal.append(
                            {
                                "agent_id": aid,
                                "mode": "live",
                                "reason": "",
                                "rebuttal": f"{aid} rebuttal",
                                "counter_points": [f"{aid} counter"],
                                "param_changes": {},
                            }
                        )
                    return {
                        "week_id": kwargs.get("week_id", "2026-W08"),
                        "rounds": [
                            {"round": 1, "phase": "opening", "speeches": opening},
                            {"round": 2, "phase": "rebuttal", "speeches": rebuttal},
                        ],
                        "moderator": {
                            "mode": "live",
                            "reason": "",
                            "summary": "live summary",
                            "consensus_actions": ["apply immediately"],
                            "risk_watch": ["watch risk"],
                        },
                        "agent_param_suggestions": suggestions,
                        "llm_warnings": [],
                    }

                with patch.dict(
                    "os.environ",
                    {
                        "AGENT_LAB_ALLOW_OFFSCHEDULE_WEEKLY": "1",
                        "AGENT_LAB_WEEKLY_APPLY_MODE": "immediate",
                    },
                ):
                    with patch.object(
                        orch.agent_engine,
                        "run_weekly_council_debate",
                        side_effect=_fake_live_weekly_debate,
                    ):
                        decision = orch.weekly_council("2026-W08")

                self.assertEqual("immediate", decision.get("promotion_apply_mode"))
                promoted_versions = decision.get("promoted_versions", {})
                self.assertEqual({"agent_a", "agent_b", "agent_c"}, set(promoted_versions.keys()))
                for aid in ["agent_a", "agent_b", "agent_c"]:
                    active = orch.registry.get_active_strategy(aid)
                    params = dict(active.get("params", {}))
                    self.assertEqual(expected_intervals[aid], int(params.get("intraday_monitor_interval_sec", 0)))
                    self.assertEqual(expected_picks[aid], int(params.get("max_daily_picks", 0)))
                    eval_row = dict((decision.get("promotion_evaluation", {}) or {}).get(aid, {}))
                    self.assertTrue(bool(eval_row.get("applied_immediately")))
                    self.assertIn(str(eval_row.get("mode", "")), {"immediate_promote", "forced_conservative_promote"})
            finally:
                orch.close()

    def test_sync_account_strict_blocks_when_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            orch = AgentLabOrchestrator(project_root=root)
            try:
                orch.init_lab(capital_krw=10_000_000, agents=1)
                fake_snapshot = {
                    "ok": True,
                    "market_scope": "ALL",
                    "source": "test",
                    "fetched_at": "2026-02-23T09:00:00+09:00",
                    "cash_krw": 1_000_000.0,
                    "equity_krw": 1_500_000.0,
                    "positions": [
                        {
                            "market": "KR",
                            "symbol": "005930",
                            "quantity": 3.0,
                            "avg_price": 70000.0,
                            "market_value_krw": 210000.0,
                            "currency": "KRW",
                            "fx_rate": 1.0,
                            "payload": {},
                        }
                    ],
                    "markets": {"KR": {}, "US": {}},
                    "errors": [],
                }
                with patch.object(orch.paper_broker, "fetch_account_snapshot", return_value=fake_snapshot):
                    out = orch.sync_account(market="ALL", strict=True)
                self.assertTrue(out.get("ok"))
                self.assertFalse(out.get("matched"))
                self.assertTrue(out.get("blocked"))
            finally:
                orch.close()

    def test_cutover_reset_archives_and_reinitializes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_date = "20260218"
            _write_signal_file(root, "KR", run_date)
            orch = AgentLabOrchestrator(project_root=root)
            try:
                orch.init_lab(capital_krw=9_000_000, agents=3)
                fake_snapshot = {
                    "ok": True,
                    "market_scope": "ALL",
                    "source": "test",
                    "fetched_at": "2026-02-23T09:00:00+09:00",
                    "cash_krw": 9_000_000.0,
                    "equity_krw": 9_000_000.0,
                    "positions": [],
                    "markets": {"KR": {}, "US": {}},
                    "errors": [],
                }
                with patch.object(orch.paper_broker, "fetch_account_snapshot", return_value=fake_snapshot):
                    payload = orch.cutover_reset(
                        require_flat=True,
                        archive=True,
                        reinit=True,
                        restart_tasks=False,
                    )
                self.assertIn("new_epoch_id", payload)
                self.assertTrue(payload.get("archive"))
                archive_root = Path(str(payload.get("archive_root", "")))
                self.assertTrue(archive_root.exists())
                agents = orch.storage.list_agents()
                self.assertEqual(3, len(agents))
            finally:
                orch.close()

    def test_unified_shadow_propose_generates_unified_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_date = "20260218"
            _write_signal_file(root, "KR", run_date)
            orch = AgentLabOrchestrator(project_root=root)
            try:
                with patch.dict(
                    os.environ,
                    {
                        "AGENT_LAB_EXECUTION_MODEL": "unified_shadow",
                        "AGENT_LAB_ENFORCE_MARKET_HOURS": "0",
                        "AGENT_LAB_SYNC_STRICT": "1",
                    },
                ):
                    orch.init_lab(capital_krw=10_000_000, agents=3)
                    orch.ingest_session(market="KR", yyyymmdd=run_date)
                    fake_snapshot = {
                        "ok": True,
                        "market_scope": "ALL",
                        "source": "test",
                        "fetched_at": "2026-02-23T09:00:00+09:00",
                        "cash_krw": 10_000_000.0,
                        "equity_krw": 10_000_000.0,
                        "positions": [],
                        "markets": {"KR": {}, "US": {}},
                        "errors": [],
                    }
                    with patch.object(orch.paper_broker, "fetch_account_snapshot", return_value=fake_snapshot):
                        with patch.object(
                            orch.paper_broker,
                            "execute_orders",
                            return_value=[
                                {
                                    "paper_order_id": 1,
                                    "symbol": "005930",
                                    "side": "BUY",
                                    "status": "FILLED",
                                    "reference_price": 70000.0,
                                    "quantity": 1.0,
                                    "fx_rate": 1.0,
                                }
                            ],
                        ):
                            payload = orch.propose_orders(market="KR", yyyymmdd=run_date)
                self.assertEqual("unified_shadow", payload.get("execution_model"))
                unified = payload.get("unified_execution", {})
                self.assertTrue(bool(unified))
                self.assertIn(unified.get("status"), {"APPROVED", "EXECUTED", "BLOCKED"})
                self.assertIn("shadow_proposals", payload)
                self.assertEqual(3, len(payload.get("shadow_proposals", [])))
            finally:
                orch.close()


if __name__ == "__main__":
    unittest.main()
