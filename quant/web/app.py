"""FastAPI control panel for botipman1000.

Exposes a lightweight management UI:
 - Dashboard: runner status, last signals, risk state
 - Config: read/edit the ``.env`` file (with secret masking)
 - Params: read/edit ``artifacts/best_params.json``
 - Backtest: trigger a one-shot backtest in a background thread, stream results
 - Optimize: trigger Optuna search in a background thread
 - Train-meta: train the meta-labeler
 - Trades: list shadow trades recorded by the current runner
 - Logs: tail the latest log file
 - Artifacts: list files under ``artifacts/`` + download

All mutating routes require HTTP basic auth when ``WEB_PASSWORD`` is set.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from quant.config import settings
from quant.live.runner import RunnerConfig
from quant.logging_setup import setup_logging
from quant.web.auth import check_auth
from quant.web.state import RUNNER_STATE, default_cfg

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
LOGS_DIR = REPO_ROOT / "logs"
DATA_DIR = REPO_ROOT / "data"
ENV_FILE = REPO_ROOT / ".env"
BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params.json"

SECRET_KEYS = {"BYBIT_API_KEY", "BYBIT_API_SECRET", "WEB_PASSWORD"}

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------- job manager
class JobManager:
    """Tracks long-running background jobs (backtest / optimize / train-meta)."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, name: str, target, *args, **kwargs) -> dict:
        with self._lock:
            cur = self._jobs.get(name)
            if cur and cur.get("status") == "running":
                return cur

            job = {
                "name": name,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "finished_at": None,
                "result": None,
                "error": None,
                "log": [],
            }
            self._jobs[name] = job

        def _runner():
            try:
                result = target(job, *args, **kwargs)
                job["result"] = result
                job["status"] = "done"
            except Exception as exc:
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["log"].append(traceback.format_exc())
                job["status"] = "error"
                logger.exception(f"job {name} failed")
            finally:
                job["finished_at"] = datetime.utcnow().isoformat()

        t = threading.Thread(target=_runner, daemon=True, name=f"job-{name}")
        t.start()
        job["thread"] = t
        return job

    def get(self, name: str) -> Optional[dict]:
        return self._jobs.get(name)

    def all(self) -> dict[str, dict]:
        return {k: _strip_thread(v) for k, v in self._jobs.items()}


def _strip_thread(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "thread"}


JOBS = JobManager()


# -------------------------------------------------------------- job targets
def _job_backtest(job: dict, symbol: str, interval: str, days: int) -> dict:
    import pandas as pd

    from quant.backtest.engine import BacktestConfig, EventBacktester
    from quant.data.bars import cusum_events
    from quant.data.loader import BybitLoader
    from quant.features.builder import FeatureParams, build_feature_matrix
    from quant.models.meta import MetaLabeler
    from quant.models.primary import PrimaryParams, PrimaryRuleModel

    job["log"].append(f"loading {symbol} {interval}m × {days}d")
    df = BybitLoader().load(symbol, interval, days=days)
    job["log"].append(f"{len(df)} rows")

    fp, pp = FeatureParams(), PrimaryParams()
    if BEST_PARAMS_PATH.exists():
        best = json.loads(BEST_PARAMS_PATH.read_text()).get("best_params", {})
        for k in FeatureParams.__dataclass_fields__:
            if k in best:
                setattr(fp, k, best[k])
        for k in PrimaryParams.__dataclass_fields__:
            if k in best:
                setattr(pp, k, best[k])
        job["log"].append("loaded best_params.json")
    feats = build_feature_matrix(df, fp).dropna()
    prim = PrimaryRuleModel(pp).compute(feats)
    side = prim["primary_side"].reindex(df.index).fillna(0)
    events = cusum_events(df["close"].loc[feats.index])
    job["log"].append(f"{len(events)} CUSUM events")

    meta_path = ARTIFACTS_DIR / "meta_model.joblib"
    meta_proba = None
    if meta_path.exists():
        ml = MetaLabeler.load(meta_path)
        X = feats.loc[events].copy()
        X["primary_score"] = prim["primary_score"].loc[events]
        proba = ml.predict_proba(X)
        meta_proba = pd.Series(proba, index=events)
        job["log"].append(f"meta model predicted on {len(X)} events")

    res = EventBacktester(BacktestConfig()).run(
        close=df["close"],
        events=events,
        side=side,
        meta_proba=meta_proba,
    )
    out_dir = ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out_dir / f"{symbol}_equity.csv")
    res.trades.to_csv(out_dir / f"{symbol}_trades.csv", index=False)
    job["log"].append(f"wrote {symbol}_equity.csv / {symbol}_trades.csv")
    return {"stats": {k: _clean_number(v) for k, v in res.stats.items()}, "n_trades": int(len(res.trades))}


def _job_optimize(job: dict, symbol: str, interval: str, days: int, trials: int) -> dict:
    from quant.data.loader import BybitLoader
    from quant.optimize.runner import OptunaRunner

    job["log"].append(f"loading {symbol} {interval}m × {days}d")
    df = BybitLoader().load(symbol, interval, days=days)
    job["log"].append(f"{len(df)} rows; running {trials} trials")
    runner = OptunaRunner(df, n_trials=trials)
    result = runner.run(study_name=f"{symbol}_{interval}m")
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save(BEST_PARAMS_PATH)
    job["log"].append(f"best value={result.best_value:.4f}")
    job["log"].append(f"saved → {BEST_PARAMS_PATH}")
    return {"best_value": float(result.best_value), "best_params": result.best_params}


def _job_train_meta(job: dict, symbol: str, interval: str, days: int) -> dict:
    import pandas as pd

    from quant.data.bars import cusum_events
    from quant.data.loader import BybitLoader
    from quant.features.builder import FeatureParams, build_feature_matrix
    from quant.labeling.triple_barrier import apply_triple_barrier, get_daily_vol
    from quant.models.meta import MetaLabeler
    from quant.models.primary import PrimaryParams, PrimaryRuleModel

    df = BybitLoader().load(symbol, interval, days=days)
    fp, pp = FeatureParams(), PrimaryParams()
    if BEST_PARAMS_PATH.exists():
        best = json.loads(BEST_PARAMS_PATH.read_text()).get("best_params", {})
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
    job["log"].append(f"{len(events)} events after primary filter")
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
    y = (labels["bin"] == 1).astype(int)
    X = feats.loc[labels.index].copy()
    X["primary_score"] = prim["primary_score"].loc[labels.index]
    job["log"].append(f"training on {len(X)} samples, positives={int(y.sum())}")
    ml = MetaLabeler()
    ml.fit(X, y)
    out = ARTIFACTS_DIR / "meta_model.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(out)
    job["log"].append(f"saved → {out}")
    return {"samples": int(len(X)), "positives": int(y.sum())}


def _job_fetch(job: dict, symbol: str, interval: str, days: int) -> dict:
    from quant.data.loader import BybitLoader

    df = BybitLoader().load(symbol, interval, days=days, refresh=True)
    job["log"].append(f"{symbol} {interval}m: {len(df)} rows")
    return {"symbol": symbol, "rows": int(len(df)), "start": str(df.index.min()), "end": str(df.index.max())}


def _clean_number(v: Any) -> Any:
    try:
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                return None
            return round(v, 6)
        return v
    except Exception:
        return str(v)


# ---------------------------------------------------------------- env helpers
def _read_env() -> list[tuple[str, str]]:
    if not ENV_FILE.exists():
        return []
    rows: list[tuple[str, str]] = []
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        rows.append((k.strip(), v.strip()))
    return rows


def _mask(key: str, val: str) -> str:
    if key.upper() in SECRET_KEYS and val:
        if len(val) <= 6:
            return "*" * len(val)
        return val[:2] + "…" + "*" * (len(val) - 4) + val[-2:]
    return val


def _write_env(pairs: list[tuple[str, str]]) -> None:
    # Preserve comments from the existing file where possible.
    existing_lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    kept_comments: list[str] = []
    for line in existing_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            kept_comments.append(line)
    body = "\n".join(f"{k}={v}" for k, v in pairs)
    header = "\n".join(kept_comments)
    out = (header + "\n" + body + "\n") if header else body + "\n"
    ENV_FILE.write_text(out)


ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _valid_env_key(key: str) -> bool:
    return bool(ENV_KEY_RE.match(key))


# -------------------------------------------------------------- FastAPI app
def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="botipman1000 control", version="1.0")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not os.getenv("WEB_PASSWORD"):
        logger.warning(
            "WEB_PASSWORD not set — the control panel is UNAUTHENTICATED. "
            "Set WEB_PASSWORD in your environment before exposing this to the internet."
        )

    def _render(name: str, request: Request, **ctx) -> HTMLResponse:
        ctx.setdefault("status", RUNNER_STATE.status())
        ctx.setdefault("nav", _nav_items(request))
        return templates.TemplateResponse(request, name, ctx)

    # --------------------------------------------------------------- health
    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok"

    # --------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def dashboard(request: Request) -> HTMLResponse:
        return _render(
            "dashboard.html",
            request,
            title="Dashboard",
            jobs=JOBS.all(),
            best_params_exists=BEST_PARAMS_PATH.exists(),
            meta_model_exists=(ARTIFACTS_DIR / "meta_model.joblib").exists(),
        )

    @app.get("/status.json", dependencies=[Depends(check_auth)])
    def status_json() -> JSONResponse:
        return JSONResponse({"status": RUNNER_STATE.status(), "jobs": JOBS.all()})

    # ----------------------------------------------------------- runner ops
    @app.post("/runner/start", dependencies=[Depends(check_auth)])
    def runner_start(
        mode: str = Form("shadow"),
        symbols: str = Form(""),
        interval: str = Form(""),
        poll_seconds: int = Form(30),
        min_meta_prob: float = Form(0.55),
        starting_equity: float = Form(10000.0),
    ) -> RedirectResponse:
        if mode not in ("shadow", "live"):
            raise HTTPException(400, "mode must be shadow or live")
        base = default_cfg()
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] or base.symbols
        cfg = RunnerConfig(
            symbols=syms,
            interval=interval or base.interval,
            poll_seconds=int(poll_seconds),
            shadow=(mode == "shadow"),
            params_path=base.params_path,
            meta_model_path=base.meta_model_path,
            starting_equity=float(starting_equity),
            min_meta_prob=float(min_meta_prob),
        )
        ok, msg = RUNNER_STATE.start(mode, cfg)  # type: ignore[arg-type]
        logger.info(f"web runner/start ok={ok} msg={msg}")
        return RedirectResponse("/", status_code=303)

    @app.post("/runner/stop", dependencies=[Depends(check_auth)])
    def runner_stop() -> RedirectResponse:
        RUNNER_STATE.stop()
        return RedirectResponse("/", status_code=303)

    @app.post("/runner/reset-halt", dependencies=[Depends(check_auth)])
    def runner_reset_halt() -> RedirectResponse:
        r = RUNNER_STATE.runner
        if r and r.risk:
            r.risk.reset_halt()
        return RedirectResponse("/", status_code=303)

    # ----------------------------------------------------------- config (.env)
    @app.get("/config", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def config_page(request: Request) -> HTMLResponse:
        rows = _read_env()
        masked = [(k, _mask(k, v)) for k, v in rows]
        return _render("config.html", request, title="Config", rows=masked)

    @app.post("/config", dependencies=[Depends(check_auth)])
    async def config_save(request: Request) -> RedirectResponse:
        form = await request.form()
        current = dict(_read_env())
        new_pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        # Existing keys kept in order
        for k, _ in _read_env():
            if k in seen:
                continue
            val_field = form.get(f"val::{k}")
            if val_field is None:
                new_pairs.append((k, current.get(k, "")))
            else:
                val = str(val_field)
                if k.upper() in SECRET_KEYS and val.strip() == "":
                    val = current.get(k, "")  # don't wipe secret on empty submit
                new_pairs.append((k, val))
            seen.add(k)
        # New rows (new_key::N / new_val::N)
        new_indices = sorted({f.split("::", 1)[1] for f in form.keys() if f.startswith("new_key::")})
        for idx in new_indices:
            k = str(form.get(f"new_key::{idx}", "")).strip().upper()
            v = str(form.get(f"new_val::{idx}", "")).strip()
            if not k or not _valid_env_key(k) or k in seen:
                continue
            new_pairs.append((k, v))
            seen.add(k)
        _write_env(new_pairs)
        logger.info(f"web: .env updated ({len(new_pairs)} keys)")
        return RedirectResponse("/config?saved=1", status_code=303)

    # ----------------------------------------------------------- params JSON
    @app.get("/params", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def params_page(request: Request) -> HTMLResponse:
        text = BEST_PARAMS_PATH.read_text() if BEST_PARAMS_PATH.exists() else "{}"
        return _render(
            "params.html",
            request,
            title="Best Params",
            text=text,
            path=str(BEST_PARAMS_PATH),
            exists=BEST_PARAMS_PATH.exists(),
        )

    @app.post("/params", dependencies=[Depends(check_auth)])
    def params_save(content: str = Form(...)) -> RedirectResponse:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"invalid JSON: {exc}")
        BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        BEST_PARAMS_PATH.write_text(json.dumps(parsed, indent=2))
        return RedirectResponse("/params?saved=1", status_code=303)

    # ----------------------------------------------------------- jobs
    @app.get("/backtest", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def backtest_page(request: Request) -> HTMLResponse:
        return _render("backtest.html", request, title="Backtest", job=JOBS.get("backtest"))

    @app.post("/backtest", dependencies=[Depends(check_auth)])
    def backtest_run(
        symbol: str = Form("BTCUSDT"),
        interval: str = Form("5"),
        days: int = Form(365),
    ) -> RedirectResponse:
        JOBS.start("backtest", _job_backtest, symbol.strip().upper(), interval.strip(), int(days))
        return RedirectResponse("/backtest", status_code=303)

    @app.get("/optimize", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def optimize_page(request: Request) -> HTMLResponse:
        return _render("optimize.html", request, title="Optimize", job=JOBS.get("optimize"))

    @app.post("/optimize", dependencies=[Depends(check_auth)])
    def optimize_run(
        symbol: str = Form("BTCUSDT"),
        interval: str = Form("5"),
        days: int = Form(365),
        trials: int = Form(100),
    ) -> RedirectResponse:
        JOBS.start(
            "optimize",
            _job_optimize,
            symbol.strip().upper(),
            interval.strip(),
            int(days),
            int(trials),
        )
        return RedirectResponse("/optimize", status_code=303)

    @app.get("/train-meta", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def train_meta_page(request: Request) -> HTMLResponse:
        return _render("train_meta.html", request, title="Train Meta", job=JOBS.get("train-meta"))

    @app.post("/train-meta", dependencies=[Depends(check_auth)])
    def train_meta_run(
        symbol: str = Form("BTCUSDT"),
        interval: str = Form("5"),
        days: int = Form(365),
    ) -> RedirectResponse:
        JOBS.start("train-meta", _job_train_meta, symbol.strip().upper(), interval.strip(), int(days))
        return RedirectResponse("/train-meta", status_code=303)

    @app.get("/fetch", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def fetch_page(request: Request) -> HTMLResponse:
        return _render("fetch.html", request, title="Fetch Data", job=JOBS.get("fetch"))

    @app.post("/fetch", dependencies=[Depends(check_auth)])
    def fetch_run(
        symbol: str = Form("BTCUSDT"),
        interval: str = Form("5"),
        days: int = Form(365),
    ) -> RedirectResponse:
        JOBS.start("fetch", _job_fetch, symbol.strip().upper(), interval.strip(), int(days))
        return RedirectResponse("/fetch", status_code=303)

    @app.get("/jobs/{name}.json", dependencies=[Depends(check_auth)])
    def job_status(name: str) -> JSONResponse:
        job = JOBS.get(name)
        if job is None:
            raise HTTPException(404, "no such job")
        return JSONResponse(_strip_thread(job))

    # ----------------------------------------------------------- trades
    @app.get("/trades", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def trades_page(request: Request) -> HTMLResponse:
        r = RUNNER_STATE.runner
        trades: list[dict] = []
        if r is not None:
            raw = list(getattr(r, "shadow_trades", []) or [])
            for t in raw[-500:]:
                trades.append({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in t.items()})
        return _render("trades.html", request, title="Trades", trades=trades)

    # ----------------------------------------------------------- logs
    @app.get("/logs", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def logs_page(request: Request, tail: int = 500) -> HTMLResponse:
        files = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        content = ""
        picked: Optional[Path] = None
        if files:
            picked = files[0]
            try:
                lines = picked.read_text(errors="replace").splitlines()
                content = "\n".join(lines[-int(tail):])
            except Exception as exc:
                content = f"<failed to read log: {exc}>"
        return _render(
            "logs.html",
            request,
            title="Logs",
            files=[f.name for f in files],
            picked=picked.name if picked else None,
            content=content,
            tail=tail,
        )

    # ----------------------------------------------------------- artifacts
    @app.get("/artifacts", response_class=HTMLResponse, dependencies=[Depends(check_auth)])
    def artifacts_page(request: Request) -> HTMLResponse:
        entries = []
        if ARTIFACTS_DIR.exists():
            for p in sorted(ARTIFACTS_DIR.iterdir()):
                if p.is_file():
                    entries.append({
                        "name": p.name,
                        "size": p.stat().st_size,
                        "mtime": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat(),
                    })
        return _render("artifacts.html", request, title="Artifacts", entries=entries)

    @app.get("/artifacts/{name}", dependencies=[Depends(check_auth)])
    def artifacts_download(name: str) -> FileResponse:
        # prevent path traversal
        p = (ARTIFACTS_DIR / name).resolve()
        if not str(p).startswith(str(ARTIFACTS_DIR.resolve())):
            raise HTTPException(400, "bad name")
        if not p.exists() or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p, filename=p.name)

    return app


def _nav_items(request: Request) -> list[dict]:
    items = [
        ("/", "Dashboard"),
        ("/config", "Config"),
        ("/params", "Params"),
        ("/fetch", "Fetch"),
        ("/optimize", "Optimize"),
        ("/train-meta", "Train Meta"),
        ("/backtest", "Backtest"),
        ("/trades", "Trades"),
        ("/logs", "Logs"),
        ("/artifacts", "Artifacts"),
    ]
    cur = request.url.path
    return [{"href": h, "label": l, "active": (cur == h or (h != "/" and cur.startswith(h)))} for h, l in items]


# module-level factory for uvicorn --factory
app = create_app()
