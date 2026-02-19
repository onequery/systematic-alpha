from __future__ import annotations

import importlib
import os
from datetime import datetime
from typing import Any, Dict, List

from systematic_alpha.agent_lab.storage import AgentLabStorage
from systematic_alpha.credentials import load_credentials
from systematic_alpha.mojito_loader import import_mojito_module


class PaperBroker:
    def __init__(self, storage: AgentLabStorage):
        self.storage = storage
        self.execution_mode = str(os.getenv("AGENT_LAB_EXECUTION_MODE", "mojito_mock")).strip().lower()
        self.us_exchange = str(os.getenv("AGENT_LAB_US_EXCHANGE", "NASD")).strip().upper() or "NASD"
        self._brokers: Dict[str, Any] = {}
        self._creds_loaded = False
        self._creds: Dict[str, Any] = {}

    def _load_creds(self) -> None:
        if self._creds_loaded:
            return
        self._creds_loaded = True
        try:
            key, secret, acc_no, _ = load_credentials(None)
            self._creds = {"key": key, "secret": secret, "acc_no": acc_no}
        except Exception:
            self._creds = {}

    def _exchange_label(self, market: str) -> str:
        # self-heal:us-exchange-resolver-v1
        if market.upper() == "KR":
            return "KR"

        normalized = {
            "NASD": "NASD",
            "NASDAQ": "NASD",
            "NYSE": "NYSE",
            "NYS": "NYSE",
            "AMEX": "AMEX",
            "AMS": "AMEX",
        }.get(self.us_exchange.upper(), self.us_exchange.upper())

        try:
            mojito = import_mojito_module()
            ki = importlib.import_module(mojito.__name__ + ".koreainvestment")
            exchange_code3 = getattr(ki, "EXCHANGE_CODE3", {})
            if isinstance(exchange_code3, dict):
                for label, code in exchange_code3.items():
                    if str(code).upper() == normalized:
                        return str(label)
        except Exception:
            pass

        fallback = {
            "NASD": "나스닥",
            "NYSE": "뉴욕",
            "AMEX": "아멕스",
        }
        return fallback.get(normalized, "나스닥")

    def _get_broker(self, market: str):
        key = market.upper()
        if key in self._brokers:
            return self._brokers[key]
        self._load_creds()
        if not self._creds:
            return None
        try:
            mojito = import_mojito_module()
            if key == "KR":
                # Domestic mock account uses default exchange routing.
                broker = mojito.KoreaInvestment(
                    api_key=self._creds["key"],
                    api_secret=self._creds["secret"],
                    acc_no=self._creds["acc_no"],
                    mock=True,
                )
            else:
                broker = mojito.KoreaInvestment(
                    api_key=self._creds["key"],
                    api_secret=self._creds["secret"],
                    acc_no=self._creds["acc_no"],
                    exchange=self._exchange_label(key),
                    mock=True,
                )
            self._brokers[key] = broker
            return broker
        except Exception:
            return None

    def _send_order(self, market: str, order: Dict[str, Any]) -> Dict[str, Any]:
        broker = self._get_broker(market)
        if broker is None:
            return {"ok": False, "reason": "broker_unavailable"}
        symbol = str(order["symbol"])
        qty = int(float(order["quantity"]))
        order_type = str(order.get("order_type", "MARKET")).upper()
        side = str(order.get("side", "BUY")).upper()
        try:
            if order_type == "LIMIT":
                px = int(float(order.get("limit_price") or order.get("reference_price") or 0))
                if side == "BUY":
                    resp = broker.create_limit_buy_order(symbol, px, qty)
                else:
                    resp = broker.create_limit_sell_order(symbol, px, qty)
            else:
                if side == "BUY":
                    resp = broker.create_market_buy_order(symbol, qty)
                else:
                    resp = broker.create_market_sell_order(symbol, qty)
            return {"ok": True, "response": resp}
        except Exception as exc:
            return {"ok": False, "reason": repr(exc)}

    def execute_orders(
        self,
        *,
        proposal_id: int,
        agent_id: str,
        market: str,
        orders: List[Dict[str, Any]],
        fx_rate: float,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        now_ts = datetime.now().isoformat(timespec="seconds")
        for order in orders:
            symbol = str(order["symbol"])
            side = str(order.get("side", "BUY")).upper()
            qty = float(order.get("quantity", 0.0) or 0.0)
            order_type = str(order.get("order_type", "MARKET")).upper()
            limit_price = order.get("limit_price")
            ref_price = float(order.get("reference_price", 0.0) or 0.0)
            if qty <= 0 or ref_price <= 0:
                continue

            broker_order_id = ""
            broker_resp: Dict[str, Any] = {}
            status = "FILLED"
            if self.execution_mode in {"mojito_mock", "mock_api"}:
                send = self._send_order(market=market, order=order)
                broker_resp = send
                if send.get("ok"):
                    payload = send.get("response")
                    if isinstance(payload, dict):
                        broker_order_id = str(payload.get("ODNO") or payload.get("output", {}).get("ODNO") or "")
                else:
                    status = "REJECTED"

            paper_order_id = self.storage.insert_paper_order(
                proposal_id=proposal_id,
                agent_id=agent_id,
                market=market,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=qty,
                limit_price=None if limit_price is None else float(limit_price),
                reference_price=ref_price,
                status=status,
                broker_order_id=broker_order_id,
                broker_response_json=broker_resp,
                submitted_at=now_ts,
            )

            if status == "FILLED":
                fill_price = ref_price if order_type != "LIMIT" else float(limit_price or ref_price)
                fill_value_krw = float(fill_price) * float(qty) * float(fx_rate)
                self.storage.insert_paper_fill(
                    paper_order_id=paper_order_id,
                    fill_price=float(fill_price),
                    fill_quantity=float(qty),
                    fill_value_krw=float(fill_value_krw),
                    fx_rate=float(fx_rate),
                    filled_at=now_ts,
                )
            results.append(
                {
                    "paper_order_id": paper_order_id,
                    "symbol": symbol,
                    "side": side,
                    "status": status,
                    "reference_price": ref_price,
                    "quantity": qty,
                    "fx_rate": fx_rate,
                }
            )
        return results
