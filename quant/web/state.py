"""Global runner state held by the web app.

The web process owns zero or one background runner (shadow or live) at a time.
Everything is threaded, not multiprocess, so the FastAPI handlers can ask the
runner for its latest signal, last tick, etc. without IPC.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal, Optional

from loguru import logger

from quant.config import settings
from quant.live.runner import LiveRunner, RunnerConfig, ShadowRunner, _BaseRunner

RunnerMode = Literal["shadow", "live"]


class RunnerState:
    """Singleton wrapper around the currently-active runner thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runner: Optional[_BaseRunner] = None
        self._mode: Optional[RunnerMode] = None
        self._last_cfg: Optional[RunnerConfig] = None

    # ---------------------------------------------------------------- getters
    @property
    def runner(self) -> Optional[_BaseRunner]:
        return self._runner

    @property
    def mode(self) -> Optional[RunnerMode]:
        return self._mode

    @property
    def running(self) -> bool:
        return self._runner is not None and self._runner.running

    def status(self) -> dict:
        r = self._runner
        if r is None:
            return {
                "running": False,
                "mode": None,
                "symbols": [],
                "last_tick": None,
                "last_error": None,
                "last_signal": {},
                "shadow_trades": 0,
                "equity": None,
                "risk_state": None,
            }
        return {
            "running": r.running,
            "mode": self._mode,
            "symbols": r.cfg.symbols,
            "interval": r.cfg.interval,
            "poll_seconds": r.cfg.poll_seconds,
            "last_tick": r.last_tick.isoformat() if r.last_tick else None,
            "last_error": r.last_error,
            "last_signal": {k: _jsonable(v) for k, v in r.last_signal.items()},
            "shadow_trades": len(getattr(r, "shadow_trades", []) or []),
            "equity": round(r.risk.state.equity, 4) if r.risk else None,
            "risk_state": _risk_jsonable(r.risk) if r.risk else None,
        }

    # ---------------------------------------------------------------- control
    def start(self, mode: RunnerMode, cfg: RunnerConfig) -> tuple[bool, str]:
        with self._lock:
            if self._runner is not None and self._runner.running:
                return False, f"{self._mode} runner already running"
            if mode == "live":
                if os.getenv("CONFIRM_LIVE", "").strip().lower() != "yes":
                    return False, "CONFIRM_LIVE=yes is required to start a live runner"
                if not (settings.bybit_api_key and settings.bybit_api_secret):
                    return False, "BYBIT_API_KEY and BYBIT_API_SECRET must be set"
                self._runner = LiveRunner(cfg)
            else:
                self._runner = ShadowRunner(cfg)
            self._mode = mode
            self._last_cfg = cfg
            ok = self._runner.start()
            if not ok:
                return False, "runner failed to start thread"
            logger.info(f"web: started {mode} runner on {cfg.symbols}")
            return True, f"{mode} runner started"

    def stop(self, timeout: float = 5.0) -> tuple[bool, str]:
        with self._lock:
            if self._runner is None or not self._runner.running:
                return False, "no runner running"
            self._runner.stop(timeout=timeout)
            mode = self._mode
            logger.info(f"web: stopped {mode} runner")
            # keep the instance around so the UI can still show final state
            return True, f"{mode} runner stopped"

    def clear(self) -> None:
        with self._lock:
            if self._runner and self._runner.running:
                self._runner.stop(timeout=3.0)
            self._runner = None
            self._mode = None


def _jsonable(sig: dict) -> dict:
    """Convert a runner signal dict to something safe for JSON/template."""
    out: dict = {}
    for k, v in sig.items():
        try:
            if hasattr(v, "isoformat"):
                out[k] = v.isoformat()
            elif isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
        except Exception:
            out[k] = str(v)
    return out


def _risk_jsonable(risk) -> dict:
    s = risk.state
    return {
        "equity": round(s.equity, 4),
        "peak_equity": round(s.peak_equity, 4),
        "daily_pnl": round(s.daily_pnl, 4),
        "drawdown": round(s.drawdown, 4),
        "halted": bool(s.halted),
        "halt_reason": s.halt_reason,
        "size_multiplier": round(s.size_multiplier, 3),
        "open_positions": len(risk.open_positions or {}),
    }


# Global singleton
RUNNER_STATE = RunnerState()


def default_cfg() -> RunnerConfig:
    symbols = [s.strip().upper() for s in (os.getenv("SYMBOLS") or ",".join(settings.symbols)).split(",") if s.strip()]
    return RunnerConfig(
        symbols=symbols,
        interval=os.getenv("INTERVAL", settings.base_timeframe),
        poll_seconds=int(os.getenv("POLL_SECONDS", "30")),
        shadow=True,
        params_path=os.getenv("PARAMS_PATH", str(Path("artifacts/best_params.json"))),
        meta_model_path=os.getenv("META_PATH", str(Path("artifacts/meta_model.joblib"))),
        starting_equity=float(os.getenv("STARTING_EQUITY", "10000")),
        min_meta_prob=float(os.getenv("MIN_META_PROB", "0.55")),
    )
