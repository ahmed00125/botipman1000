"""Bybit historical kline loader with parquet cache.

Uses pybit v5 unified_trading.HTTP when available. Falls back to the public
REST endpoint so the module is usable without credentials for backtesting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from quant.config import settings

BYBIT_REST = "https://api.bybit.com"
BYBIT_TESTNET_REST = "https://api-testnet.bybit.com"


@dataclass
class Kline:
    start: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


class BybitLoader:
    """Thin REST loader for Bybit v5 klines with local parquet cache."""

    def __init__(self, testnet: bool | None = None, category: str | None = None):
        self.testnet = settings.bybit_testnet if testnet is None else testnet
        self.category = category or settings.category
        self.base_url = BYBIT_TESTNET_REST if self.testnet else BYBIT_REST
        self.cache_dir = settings.data_dir / "raw"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ API
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Page through /v5/market/kline until [start_ms, end_ms] is covered."""
        all_rows: list[list] = []
        cursor = start_ms
        url = f"{self.base_url}/v5/market/kline"
        while cursor < end_ms:
            params = {
                "category": self.category,
                "symbol": symbol,
                "interval": interval,
                "start": cursor,
                "end": end_ms,
                "limit": limit,
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                payload = r.json()
            except Exception as exc:  # network hiccups
                logger.warning(f"kline fetch failed, retrying in 2s: {exc}")
                time.sleep(2)
                continue

            if payload.get("retCode") != 0:
                raise RuntimeError(f"Bybit error: {payload}")

            rows = payload.get("result", {}).get("list", []) or []
            if not rows:
                break
            # Bybit returns newest-first; reverse.
            rows = list(reversed(rows))
            all_rows.extend(rows)
            last_start = int(rows[-1][0])
            interval_ms = self._interval_ms(interval)
            next_cursor = last_start + interval_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            time.sleep(0.08)  # be polite

        if not all_rows:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "turnover"]
            )

        df = pd.DataFrame(
            all_rows,
            columns=["start", "open", "high", "low", "close", "volume", "turnover"],
        )
        df["start"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            df[c] = df[c].astype(float)
        df = df.drop_duplicates(subset=["start"]).sort_values("start")
        df = df.set_index("start")
        return df

    @staticmethod
    def _interval_ms(interval: str) -> int:
        mapping = {
            "1": 60_000,
            "3": 3 * 60_000,
            "5": 5 * 60_000,
            "15": 15 * 60_000,
            "30": 30 * 60_000,
            "60": 60 * 60_000,
            "120": 2 * 60 * 60_000,
            "240": 4 * 60 * 60_000,
            "360": 6 * 60 * 60_000,
            "720": 12 * 60 * 60_000,
            "D": 24 * 60 * 60_000,
        }
        return mapping.get(interval, 60_000)

    # ---------------------------------------------------------------- Cache
    def cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol}_{interval}.parquet"

    def load(
        self,
        symbol: str,
        interval: str,
        days: int = 365,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return cached frame (+ incremental update unless refresh=True)."""
        path = self.cache_path(symbol, interval)
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = now_ms - days * 24 * 60 * 60 * 1000

        if path.exists() and not refresh:
            cached = pd.read_parquet(path)
            if len(cached):
                last = int(cached.index[-1].timestamp() * 1000) + self._interval_ms(
                    interval
                )
                if last < now_ms:
                    logger.info(f"{symbol} {interval}m: incremental fetch from {last}")
                    fresh = self.fetch_klines(symbol, interval, last, now_ms)
                    if len(fresh):
                        cached = pd.concat([cached, fresh]).drop_duplicates()
                cached = cached.sort_index()
                cached = cached[cached.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC")]
                cached.to_parquet(path)
                return cached

        logger.info(f"{symbol} {interval}m: cold fetch {days}d")
        df = self.fetch_klines(symbol, interval, start_ms, now_ms)
        df.to_parquet(path)
        return df
