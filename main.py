"""Quant CLI — fetch / backtest / optimize / shadow / live."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger
from rich import print as rprint
from rich.table import Table

from quant.config import settings
from quant.logging_setup import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)


# ----------------------------------------------------------------------- fetch
@app.command()
def fetch(
    symbol: str = typer.Option("BTCUSDT", help="Bybit symbol"),
    interval: str = typer.Option("5", help="Kline interval in minutes or D"),
    days: int = typer.Option(365, help="History days"),
    refresh: bool = typer.Option(False, help="Force refetch"),
):
    """Download historical klines into parquet cache."""
    setup_logging()
    from quant.data.loader import BybitLoader

    df = BybitLoader().load(symbol, interval, days=days, refresh=refresh)
    rprint(f"[green]{symbol}[/] {interval}m: {len(df)} rows  ({df.index.min()} → {df.index.max()})")


# --------------------------------------------------------------------- backtest
@app.command()
def backtest(
    symbol: str = typer.Option("BTCUSDT"),
    interval: str = typer.Option("5"),
    days: int = typer.Option(365),
    params: str = typer.Option(None, help="Path to best_params.json"),
    meta: str = typer.Option(None, help="Path to meta model"),
):
    """Run a single backtest with current (or loaded) params."""
    setup_logging()
    from quant.backtest.engine import BacktestConfig, EventBacktester
    from quant.data.bars import cusum_events
    from quant.data.loader import BybitLoader
    from quant.features.builder import FeatureParams, build_feature_matrix
    from quant.models.meta import MetaLabeler
    from quant.models.primary import PrimaryParams, PrimaryRuleModel

    df = BybitLoader().load(symbol, interval, days=days)
    fp, pp = FeatureParams(), PrimaryParams()
    if params:
        raw = json.loads(Path(params).read_text())
        best = raw.get("best_params", raw)
        for k in FeatureParams.__dataclass_fields__:
            if k in best:
                setattr(fp, k, best[k])
        for k in PrimaryParams.__dataclass_fields__:
            if k in best:
                setattr(pp, k, best[k])

    feats = build_feature_matrix(df, fp).dropna()
    prim = PrimaryRuleModel(pp).compute(feats)
    side = prim["primary_side"]
    events = cusum_events(df["close"].loc[feats.index])

    meta_proba = None
    if meta and Path(meta).exists():
        ml = MetaLabeler.load(meta)
        proba = ml.predict_proba(feats.loc[events])
        meta_proba = dict(zip(events, proba))
        import pandas as pd
        meta_proba = pd.Series(meta_proba)

    res = EventBacktester(BacktestConfig()).run(
        close=df["close"], events=events, side=side.reindex(df.index).fillna(0),
        meta_proba=meta_proba,
    )
    table = Table(title=f"{symbol} backtest")
    table.add_column("metric")
    table.add_column("value")
    for k, v in res.stats.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    rprint(table)

    Path(settings.artifact_dir).mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(settings.artifact_dir / f"{symbol}_equity.csv")
    res.trades.to_csv(settings.artifact_dir / f"{symbol}_trades.csv", index=False)


# --------------------------------------------------------------------- optimize
@app.command()
def optimize(
    symbol: str = typer.Option("BTCUSDT"),
    interval: str = typer.Option("5"),
    days: int = typer.Option(365),
    trials: int = typer.Option(200),
    out: str = typer.Option("artifacts/best_params.json"),
):
    """Phase 1 — search best parameter combination via CPCV + Optuna."""
    setup_logging()
    from quant.data.loader import BybitLoader
    from quant.optimize.runner import OptunaRunner

    df = BybitLoader().load(symbol, interval, days=days)
    logger.info(f"optimizing on {len(df)} rows")
    runner = OptunaRunner(df, n_trials=trials)
    result = runner.run(study_name=f"{symbol}_{interval}m")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    result.save(out)
    rprint(f"[green]best value[/]: {result.best_value:.4f}")
    rprint(json.dumps(result.best_params, indent=2))


# ------------------------------------------------------------------------- meta
@app.command("train-meta")
def train_meta(
    symbol: str = typer.Option("BTCUSDT"),
    interval: str = typer.Option("5"),
    days: int = typer.Option(365),
    params: str = typer.Option("artifacts/best_params.json"),
    out: str = typer.Option("artifacts/meta_model.joblib"),
):
    """Train the meta-labeling classifier on top of the primary model."""
    setup_logging()
    import pandas as pd
    from quant.data.bars import cusum_events
    from quant.data.loader import BybitLoader
    from quant.features.builder import FeatureParams, build_feature_matrix
    from quant.labeling.triple_barrier import apply_triple_barrier, get_daily_vol
    from quant.models.meta import MetaLabeler
    from quant.models.primary import PrimaryParams, PrimaryRuleModel

    df = BybitLoader().load(symbol, interval, days=days)
    fp, pp = FeatureParams(), PrimaryParams()
    if Path(params).exists():
        best = json.loads(Path(params).read_text()).get("best_params", {})
        for k in FeatureParams.__dataclass_fields__:
            if k in best:
                setattr(fp, k, best[k])
        for k in PrimaryParams.__dataclass_fields__:
            if k in best:
                setattr(pp, k, best[k])

    feats = build_feature_matrix(df, fp).dropna()
    close = df["close"].loc[feats.index]
    prim = PrimaryRuleModel(pp).compute(feats)
    side = prim["primary_side"]
    events = cusum_events(close)
    events = events.intersection(side[side != 0].index)

    tgt = get_daily_vol(close, span=100).reindex(close.index).ffill().bfill().clip(lower=5e-4)
    labels = apply_triple_barrier(
        close=close,
        events=events,
        target_vol=tgt,
        pt_mult=2.0,
        sl_mult=1.0,
        max_hold_bars=48,
        side=side.reindex(events),
    )
    # Binary: 1 if barrier == +1 (TP), else 0
    y = (labels["bin"] == 1).astype(int)
    X = feats.loc[labels.index].copy()
    X["primary_score"] = prim["primary_score"].loc[labels.index]
    logger.info(f"training meta on {len(X)} samples, positives={y.sum()}/{len(y)}")
    ml = MetaLabeler()
    ml.fit(X, y)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ml.save(out)
    rprint(f"[green]saved meta model → {out}[/]")


# ---------------------------------------------------------------------- shadow
@app.command()
def shadow(
    symbols: str = typer.Option("BTCUSDT,ETHUSDT"),
    interval: str = typer.Option("5"),
    once: bool = typer.Option(False, help="Single pass instead of loop"),
):
    """Run the strategy in paper-trading shadow mode."""
    setup_logging()
    from quant.live.runner import RunnerConfig, ShadowRunner

    cfg = RunnerConfig(
        symbols=[s.strip().upper() for s in symbols.split(",") if s.strip()],
        interval=interval,
        shadow=True,
    )
    runner = ShadowRunner(cfg)
    if once:
        runner.run_once()
    else:
        runner.loop()


# ------------------------------------------------------------------------ live
@app.command()
def live(
    symbols: str = typer.Option("BTCUSDT,ETHUSDT"),
    interval: str = typer.Option("5"),
    confirm: bool = typer.Option(False, "--confirm", help="Required to trade real funds"),
    once: bool = typer.Option(False),
):
    """Run the strategy against real Bybit API (requires --confirm)."""
    setup_logging()
    if not confirm:
        rprint("[red]refusing to run live without --confirm[/]")
        raise typer.Exit(2)
    from quant.live.runner import LiveRunner, RunnerConfig

    cfg = RunnerConfig(
        symbols=[s.strip().upper() for s in symbols.split(",") if s.strip()],
        interval=interval,
        shadow=False,
    )
    runner = LiveRunner(cfg)
    if once:
        runner.run_once()
    else:
        runner.loop()


if __name__ == "__main__":
    app()
