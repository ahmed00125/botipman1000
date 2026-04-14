"""Central config loaded from env + defaults."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Bybit
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_testnet: bool = True
    category: str = "linear"

    # Universe
    symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
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

    @field_validator("symbols", mode="before")
    @classmethod
    def _split_symbols(cls, v):
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return v

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.artifact_dir, self.log_dir):
            p.mkdir(parents=True, exist_ok=True)
            (self.data_dir / "raw").mkdir(exist_ok=True)
            (self.data_dir / "processed").mkdir(exist_ok=True)


settings = Settings()
settings.ensure_dirs()
