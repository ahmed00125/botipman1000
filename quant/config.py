"""Central config loaded from env + defaults."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Bybit
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_testnet: bool = True
    category: str = "linear"

    # Universe — stored as a comma-separated string so env vars don't need to
    # be JSON-encoded. Use ``settings.symbols`` (property) to get the list.
    symbols_raw: str = "BTCUSDT,ETHUSDT"
    base_timeframe: str = "5"  # minutes for raw klines

    # Risk
    max_leverage: float = 3.0
    risk_per_trade: float = 0.01
    vol_target_annual: float = 0.15
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.15
    kelly_fraction: float = 0.25

    # Paths
    data_dir: Path = Path("./data")
    artifact_dir: Path = Path("./artifacts")
    log_dir: Path = Path("./logs")

    # Ops
    log_level: str = "INFO"

    @property
    def symbols(self) -> List[str]:
        return [s.strip().upper() for s in self.symbols_raw.split(",") if s.strip()]

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.artifact_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)


# Pydantic-settings reads env vars by field name; alias SYMBOLS → symbols_raw
# via environment substitution at load time.
import os  # noqa: E402

if "SYMBOLS" in os.environ and "SYMBOLS_RAW" not in os.environ:
    os.environ["SYMBOLS_RAW"] = os.environ["SYMBOLS"]

settings = Settings()
settings.ensure_dirs()
