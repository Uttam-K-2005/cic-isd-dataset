"""Environment-backed API settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ARTIFACT_DIR = Path("artifacts/models/xgboost_20261029T152352Z")


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    max_batch_size: int = 64

    @classmethod
    def from_env(cls) -> "Settings":
        artifact_dir = Path(os.environ.get("CICIDS_MODEL_ARTIFACT", str(DEFAULT_ARTIFACT_DIR)))
        try:
            max_batch_size = int(os.environ.get("CICIDS_MAX_BATCH_SIZE", "64"))
        except ValueError as exc:
            raise ValueError("CICIDS_MAX_BATCH_SIZE must be an integer") from exc
        if not 1 <= max_batch_size <= 256:
            raise ValueError("CICIDS_MAX_BATCH_SIZE must be between 1 and 256")
        return cls(artifact_dir=artifact_dir, max_batch_size=max_batch_size)
