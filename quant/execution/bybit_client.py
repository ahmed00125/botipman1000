"""Thin Bybit v5 unified-trading client wrapper.

We prefer the official ``pybit`` SDK but fall back to raw REST so the module
loads even if pybit is missing.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from loguru import logger

from quant.config import settings


@dataclass
class OrderRequest:
    symbol: str
    side: str              # "Buy" or "Sell"
    qty: float
    order_type: str = "Market"
    price: float | None = None
    reduce_only: bool = False
    time_in_force: str = "GTC"
    client_order_id: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


class BybitClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool | None = None,
        category: str | None = None,
    ):
        self.api_key = api_key or settings.bybit_api_key
        self.api_secret = api_secret or settings.bybit_api_secret
        self.testnet = settings.bybit_testnet if testnet is None else testnet
        self.category = category or settings.category
        self.base_url = (
            "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
        )
        self._use_pybit = False
        self._http = None
        try:
            from pybit.unified_trading import HTTP  # type: ignore
            self._http = HTTP(
                testnet=self.testnet,
                api_key=self.api_key or None,
                api_secret=self.api_secret or None,
            )
            self._use_pybit = True
            logger.info("BybitClient: using pybit SDK")
        except Exception as exc:
            logger.warning(f"pybit unavailable, using raw REST: {exc}")

    # ------------------------------------------------------------ signing
    def _sign(self, params: dict) -> dict:
        ts = str(int(time.time() * 1000))
        recv_window = "5000"
        query = json.dumps(params, separators=(",", ":"), sort_keys=True)
        payload = ts + self.api_key + recv_window + query
        sig = hmac.new(
            self.api_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": sig,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------ market
    def get_ticker(self, symbol: str) -> dict:
        if self._use_pybit:
            return self._http.get_tickers(category=self.category, symbol=symbol)
        r = requests.get(
            f"{self.base_url}/v5/market/tickers",
            params={"category": self.category, "symbol": symbol},
            timeout=10,
        )
        return r.json()

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> dict:
        if self._use_pybit:
            return self._http.get_wallet_balance(accountType=account_type)
        headers = self._sign({})
        r = requests.get(
            f"{self.base_url}/v5/account/wallet-balance",
            params={"accountType": account_type},
            headers=headers,
            timeout=10,
        )
        return r.json()

    # ------------------------------------------------------------ orders
    def place_order(self, req: OrderRequest) -> dict:
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                "missing API credentials — refusing to place order"
            )
        body = {
            "category": self.category,
            "symbol": req.symbol,
            "side": req.side,
            "orderType": req.order_type,
            "qty": str(req.qty),
            "timeInForce": req.time_in_force,
            "reduceOnly": req.reduce_only,
        }
        if req.price is not None:
            body["price"] = str(req.price)
        if req.client_order_id:
            body["orderLinkId"] = req.client_order_id
        if req.stop_loss is not None:
            body["stopLoss"] = str(req.stop_loss)
        if req.take_profit is not None:
            body["takeProfit"] = str(req.take_profit)

        if self._use_pybit:
            return self._http.place_order(**body)
        headers = self._sign(body)
        r = requests.post(
            f"{self.base_url}/v5/order/create",
            data=json.dumps(body),
            headers=headers,
            timeout=10,
        )
        return r.json()

    def set_leverage(self, symbol: str, leverage: float) -> dict:
        body = {
            "category": self.category,
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        if self._use_pybit:
            return self._http.set_leverage(**body)
        headers = self._sign(body)
        r = requests.post(
            f"{self.base_url}/v5/position/set-leverage",
            data=json.dumps(body),
            headers=headers,
            timeout=10,
        )
        return r.json()

    def get_positions(self, symbol: str | None = None) -> dict:
        if self._use_pybit:
            kwargs = {"category": self.category}
            if symbol:
                kwargs["symbol"] = symbol
            else:
                kwargs["settleCoin"] = "USDT"
            return self._http.get_positions(**kwargs)
        params = {"category": self.category}
        if symbol:
            params["symbol"] = symbol
        else:
            params["settleCoin"] = "USDT"
        headers = self._sign(params)
        r = requests.get(
            f"{self.base_url}/v5/position/list",
            params=params,
            headers=headers,
            timeout=10,
        )
        return r.json()

    def close_position(self, symbol: str, side: str, qty: float) -> dict:
        opp = "Sell" if side == "Buy" else "Buy"
        return self.place_order(
            OrderRequest(
                symbol=symbol,
                side=opp,
                qty=qty,
                order_type="Market",
                reduce_only=True,
                client_order_id=f"close-{symbol}-{int(time.time())}",
            )
        )
