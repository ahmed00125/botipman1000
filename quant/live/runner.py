"""Live / shadow trading runner.

Shadow mode: streams klines, computes signals, places *simulated* orders and
records them to SQLite so live stats can be compared to the backtest.

Live mode: the same flow, but submits real orders to Bybit with strict
safety checks — max position, risk halt, drift monitor. Requires --live flag.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant.config import settings
from quant.data.loader import BybitLoader
from quant.execution.bybit_client import BybitClient, OrderRequest
from quant.execution.drift import DriftMonitor
from quant.features.builder import FeatureParams, build_feature_matrix
from quant.models.meta import MetaLabeler
from quant.models.primary import PrimaryParams, PrimaryRuleModel
from quant.risk.manager import RiskLimits, RiskManager
from quant.sizing.kelly import combined_size


@dataclass
class RunnerConfig:
    symbols: list[str]
    interval: str = "5"
    poll_seconds: int = 30
    shadow: bool = True
    params_path: str = "artifacts/best_params.json"
    meta_model_path: str | None = "artifacts/meta_model.joblib"
    starting_equity: float = 10_000.0
    min_meta_prob: float = 0.55


class _BaseRunner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self.loader = BybitLoader()
        self.client: Optional[BybitClient] = None
        self.risk = RiskManager(
            RiskLimits(
                max_leverage=settings.max_leverage,
                risk_per_trade=settings.risk_per_trade,
                daily_loss_limit=settings.daily_loss_limit,
                max_drawdown=settings.max_drawdown,
            ),
            starting_equity=cfg.starting_equity,
        )
        self.feature_params, self.primary_params = self._load_params()
        self.meta: Optional[MetaLabeler] = None
        if cfg.meta_model_path and Path(cfg.meta_model_path).exists():
            try:
                self.meta, fp_saved = MetaLabeler.load(cfg.meta_model_path)
                if fp_saved is not None:
                    # Live features must match how the meta model was trained.
                    self.feature_params = fp_saved
                    logger.info("using feature params bundled in meta model")
                logger.info(f"loaded meta model from {cfg.meta_model_path}")
            except Exception as exc:
                logger.warning(f"meta model load failed: {exc}")
        self.primary = PrimaryRuleModel(self.primary_params)
        self.drift: dict[str, DriftMonitor] = {}
        self.shadow_trades: list[dict] = []
        # Threading + status
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_signal: dict[str, dict] = {}
        self.last_error: str | None = None
        self.last_tick: datetime | None = None

    # --------------------------------------------------------- thread API
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.running:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_background, daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> bool:
        if not self.running:
            return False
        self._stop.set()
        self._thread.join(timeout=timeout)
        return True

    def _run_background(self) -> None:
        self._background_label()
        while not self._stop.is_set():
            try:
                self.run_once()
                self.last_tick = datetime.utcnow()
                self.last_error = None
            except Exception as exc:
                self.last_error = repr(exc)
                logger.exception(f"runner step failed: {exc}")
            self._stop.wait(self.cfg.poll_seconds)
        logger.info("runner thread exiting cleanly")

    def _background_label(self) -> None:
        logger.info(f"runner thread starting: {self.__class__.__name__}")

    def _load_params(self) -> tuple[FeatureParams, PrimaryParams]:
        path = Path(self.cfg.params_path)
        if not path.exists():
            logger.warning(f"no params at {path}, using defaults")
            return FeatureParams(), PrimaryParams()
        raw = json.loads(path.read_text())
        best = raw.get("best_params", raw)
        fp_keys = FeatureParams.__dataclass_fields__.keys()
        pp_keys = PrimaryParams.__dataclass_fields__.keys()
        fp = FeatureParams(**{k: best[k] for k in fp_keys if k in best})
        pp = PrimaryParams(**{k: best[k] for k in pp_keys if k in best})
        logger.info(f"loaded params: {fp} / {pp}")
        return fp, pp

    # --------------------------------------------------------- signal loop
    def _step(self, symbol: str) -> dict | None:
        df = self.loader.load(symbol, self.cfg.interval, days=60, refresh=True)
        if len(df) < 300:
            return None
        feats = build_feature_matrix(df, self.feature_params).dropna()
        if len(feats) < 200:
            return None
        prim = self.primary.compute(feats)
        last_ts = feats.index[-1]
        side = int(prim["primary_side"].iloc[-1])
        score = float(prim["primary_score"].iloc[-1])
        mode = int(prim["primary_mode"].iloc[-1]) if "primary_mode" in prim.columns else 0
        close = float(df["close"].iloc[-1])

        # Meta-filter
        prob = None
        if side != 0 and self.meta is not None:
            try:
                prob = float(self.meta.predict_proba(feats.tail(1))[0])
            except Exception as exc:
                logger.warning(f"meta predict failed: {exc}")
                prob = None
            if prob is not None and prob < self.cfg.min_meta_prob:
                side = 0

        # Drift monitor
        if symbol not in self.drift:
            self.drift[symbol] = DriftMonitor(feats["log_ret"].dropna().tail(1000))
        dr = self.drift[symbol].update(float(feats["log_ret"].iloc[-1]))
        if dr.get("drift"):
            logger.warning(f"{symbol} drift detected: {dr}")
            return {"symbol": symbol, "action": "halt-drift", "ts": last_ts}

        sig = {
            "symbol": symbol,
            "ts": last_ts,
            "side": side,
            "score": score,
            "mode": mode,
            "meta_prob": prob,
            "price": close,
            "atr": float(feats["atr_14"].iloc[-1]),
            "vol_64": float(feats["vol_64"].iloc[-1]),
        }
        self.last_signal[symbol] = sig
        return sig


class ShadowRunner(_BaseRunner):
    def run_once(self) -> list[dict]:
        results = []
        for sym in self.cfg.symbols:
            sig = self._step(sym)
            if sig is None:
                continue
            if sig.get("action") == "halt-drift":
                continue
            if sig["side"] != 0:
                sig["action"] = "paper_fill"
                self.shadow_trades.append(sig)
                logger.info(
                    f"[SHADOW] {sig['symbol']} side={sig['side']} "
                    f"score={sig['score']:.3f} prob={sig.get('meta_prob')} @ {sig['price']}"
                )
            results.append(sig)
        return results

    def loop(self) -> None:
        logger.info("Shadow loop starting — no real orders will be placed")
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.exception(f"shadow step failed: {exc}")
            time.sleep(self.cfg.poll_seconds)


class LiveRunner(_BaseRunner):
    def __init__(self, cfg: RunnerConfig):
        super().__init__(cfg)
        self.client = BybitClient()

    def _open(self, sig: dict) -> None:
        symbol = sig["symbol"]
        side = sig["side"]
        price = sig["price"]
        atr = sig["atr"]
        mode = sig.get("mode", 1)
        if side == 0:
            return
        ok, reason = self.risk.can_open(symbol, side)
        if not ok:
            logger.info(f"risk blocked {symbol}: {reason}")
            return

        vol_ann = sig["vol_64"] * np.sqrt(365 * 24 * 12)
        prob = sig.get("meta_prob") or 0.55
        # Regime-aware risk/reward — matches the backtest engine's barriers.
        if mode == 2:  # range-fade
            sl_atr_mult, pt_atr_mult = 1.0, 1.5
        else:  # trend-follow (default)
            sl_atr_mult, pt_atr_mult = 1.5, 3.0
        wlr = pt_atr_mult / sl_atr_mult
        qty = combined_size(
            equity=self.risk.state.equity,
            price=price,
            realized_vol_ann=vol_ann,
            win_prob=prob,
            win_loss_ratio=wlr,
            target_vol_ann=settings.vol_target_ann,
            max_leverage=settings.max_leverage,
            kelly_fraction=settings.kelly_fraction,
            min_prob=self.cfg.min_meta_prob,
        )
        qty = round(qty, 4)
        if qty <= 0:
            return

        sl_px = price - side * sl_atr_mult * atr
        tp_px = price + side * pt_atr_mult * atr
        req = OrderRequest(
            symbol=symbol,
            side="Buy" if side > 0 else "Sell",
            qty=qty,
            order_type="Market",
            stop_loss=round(sl_px, 4),
            take_profit=round(tp_px, 4),
            client_order_id=f"quant-{symbol}-{int(time.time())}",
        )
        try:
            resp = self.client.place_order(req)
            logger.info(f"[LIVE] placed {req}: {resp}")
            self.risk.register_open(symbol, side, qty, price)
        except Exception as exc:
            logger.error(f"order failed: {exc}")

    def run_once(self) -> None:
        for sym in self.cfg.symbols:
            sig = self._step(sym)
            if sig is None or sig.get("action") == "halt-drift":
                continue
            if sig["side"] != 0:
                self._open(sig)

    def loop(self) -> None:
        logger.warning("LIVE loop starting — real orders will be placed")
        while True:
            try:
                self.run_once()
            except Exception as exc:
                logger.exception(f"live step failed: {exc}")
            time.sleep(self.cfg.poll_seconds)
