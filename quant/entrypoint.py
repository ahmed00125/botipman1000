"""Single process entry-point for container/Railway deployments.

Reads ``RUN_MODE`` from env and dispatches to the appropriate subsystem:

    RUN_MODE=web      → FastAPI control panel (default)
    RUN_MODE=shadow   → paper trading loop
    RUN_MODE=live     → real orders (requires CONFIRM_LIVE=yes)
    RUN_MODE=optimize → one-shot parameter search, exits when done
    RUN_MODE=backtest → one-shot backtest with loaded params, exits
    RUN_MODE=fetch    → one-shot historical fetch, exits

Additional env:
    SYMBOLS            comma-separated (default from config)
    INTERVAL           e.g. 5, 15 (default from config)
    DAYS               history window for fetch/optimize/backtest (default 365)
    TRIALS             optimizer trials (default 200)
    PARAMS_PATH        best params path (default artifacts/best_params.json)
    META_PATH          meta model path (default artifacts/meta_model.joblib)
    POLL_SECONDS       shadow/live loop interval (default 30)
    CONFIRM_LIVE       must be "yes" to run RUN_MODE=live
    PORT               port for RUN_MODE=web (default 8000, Railway sets this)
    WEB_HOST           bind host for RUN_MODE=web (default 0.0.0.0)
    WEB_PASSWORD       enable HTTP basic auth on the panel
    WEB_USERNAME       basic-auth user (default admin)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from loguru import logger

from quant.config import settings
from quant.logging_setup import setup_logging


def _get_symbols() -> list[str]:
    raw = os.getenv("SYMBOLS") or ",".join(settings.symbols)
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _run_fetch() -> int:
    from quant.data.loader import BybitLoader

    loader = BybitLoader()
    interval = os.getenv("INTERVAL", settings.base_timeframe)
    days = int(os.getenv("DAYS", "365"))
    for sym in _get_symbols():
        df = loader.load(sym, interval, days=days, refresh=True)
        logger.info(f"{sym} {interval}m: {len(df)} rows")
    return 0


def _run_optimize() -> int:
    from quant.data.loader import BybitLoader
    from quant.optimize.runner import OptunaRunner

    symbol = _get_symbols()[0]
    interval = os.getenv("INTERVAL", settings.base_timeframe)
    days = int(os.getenv("DAYS", "365"))
    trials = int(os.getenv("TRIALS", "200"))
    out = Path(os.getenv("PARAMS_PATH", settings.artifact_dir / "best_params.json"))

    df = BybitLoader().load(symbol, interval, days=days)
    logger.info(f"optimizing {symbol} on {len(df)} rows with {trials} trials")
    runner = OptunaRunner(df, n_trials=trials)
    result = runner.run(study_name=f"{symbol}_{interval}m")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)
    logger.info(f"best value={result.best_value:.4f}")
    logger.info(f"saved params → {out}")
    return 0


def _run_backtest() -> int:
    import pandas as pd
    from quant.backtest.engine import BacktestConfig, EventBacktester
    from quant.data.bars import cusum_events
    from quant.data.loader import BybitLoader
    from quant.features.builder import FeatureParams, build_feature_matrix
    from quant.models.meta import MetaLabeler
    from quant.models.primary import PrimaryParams, PrimaryRuleModel

    symbol = _get_symbols()[0]
    interval = os.getenv("INTERVAL", settings.base_timeframe)
    days = int(os.getenv("DAYS", "365"))
    params_path = Path(os.getenv("PARAMS_PATH", settings.artifact_dir / "best_params.json"))
    meta_path = Path(os.getenv("META_PATH", settings.artifact_dir / "meta_model.joblib"))

    df = BybitLoader().load(symbol, interval, days=days)
    fp, pp = FeatureParams(), PrimaryParams()
    if params_path.exists():
        best = json.loads(params_path.read_text()).get("best_params", {})
        for k in FeatureParams.__dataclass_fields__:
            if k in best:
                setattr(fp, k, best[k])
        for k in PrimaryParams.__dataclass_fields__:
            if k in best:
                setattr(pp, k, best[k])

    ml = None
    if meta_path.exists():
        ml, fp_saved = MetaLabeler.load(meta_path)
        if fp_saved is not None:
            fp = fp_saved
            logger.info("using feature params bundled in meta model")

    feats = build_feature_matrix(df, fp).dropna()
    prim = PrimaryRuleModel(pp).compute(feats)
    side = prim["primary_side"].reindex(df.index).fillna(0)
    events = cusum_events(df["close"].loc[feats.index])
    meta_proba = None
    if ml is not None:
        X = feats.loc[events].copy()
        X["primary_score"] = prim["primary_score"].loc[events]
        proba = ml.predict_proba(X)
        meta_proba = pd.Series(proba, index=events)
    res = EventBacktester(BacktestConfig()).run(
        close=df["close"], events=events, side=side, meta_proba=meta_proba
    )
    logger.info(f"backtest stats: {res.stats}")
    out_dir = settings.artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out_dir / f"{symbol}_equity.csv")
    res.trades.to_csv(out_dir / f"{symbol}_trades.csv", index=False)
    return 0


def _run_shadow() -> int:
    from quant.live.runner import RunnerConfig, ShadowRunner

    cfg = RunnerConfig(
        symbols=_get_symbols(),
        interval=os.getenv("INTERVAL", settings.base_timeframe),
        shadow=True,
        poll_seconds=int(os.getenv("POLL_SECONDS", "30")),
        params_path=os.getenv("PARAMS_PATH", "artifacts/best_params.json"),
        meta_model_path=os.getenv("META_PATH", "artifacts/meta_model.joblib"),
    )
    ShadowRunner(cfg).loop()
    return 0


def _run_live() -> int:
    if os.getenv("CONFIRM_LIVE", "").strip().lower() != "yes":
        logger.error("RUN_MODE=live requires CONFIRM_LIVE=yes")
        return 2
    if not (settings.bybit_api_key and settings.bybit_api_secret):
        logger.error("BYBIT_API_KEY and BYBIT_API_SECRET are required for live mode")
        return 2
    from quant.live.runner import LiveRunner, RunnerConfig

    cfg = RunnerConfig(
        symbols=_get_symbols(),
        interval=os.getenv("INTERVAL", settings.base_timeframe),
        shadow=False,
        poll_seconds=int(os.getenv("POLL_SECONDS", "30")),
        params_path=os.getenv("PARAMS_PATH", "artifacts/best_params.json"),
        meta_model_path=os.getenv("META_PATH", "artifacts/meta_model.joblib"),
    )
    logger.warning(
        f"LIVE MODE starting on {cfg.symbols} testnet={settings.bybit_testnet}"
    )
    LiveRunner(cfg).loop()
    return 0


def _run_web() -> int:
    import uvicorn

    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"web control panel starting on {host}:{port}")
    if not os.getenv("WEB_PASSWORD"):
        logger.warning("WEB_PASSWORD is not set — the panel is UNAUTHENTICATED")
    uvicorn.run("quant.web.app:app", host=host, port=port, log_level=settings.log_level.lower())
    return 0


MODES = {
    "fetch": _run_fetch,
    "optimize": _run_optimize,
    "backtest": _run_backtest,
    "shadow": _run_shadow,
    "live": _run_live,
    "web": _run_web,
}


def main() -> int:
    setup_logging()
    mode = os.getenv("RUN_MODE", "web").strip().lower()
    if mode not in MODES:
        logger.error(f"unknown RUN_MODE={mode}. valid: {list(MODES)}")
        return 2
    logger.info(f"entrypoint: RUN_MODE={mode}, testnet={settings.bybit_testnet}")
    try:
        return MODES[mode]()
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    except Exception:
        logger.exception("entrypoint failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
