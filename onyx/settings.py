"""运行期路径与配置解析。

只依赖 stdlib；数据目录可用 ONYX_DATA_DIR 覆盖，便于测试与多实例并存。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR_NAME = ".data"


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "onyx.sqlite"

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def dataset_cache_dir(self) -> Path:
        return self.cache_dir / "datasets"

    def ensure_dirs(self) -> Settings:
        for p in (self.data_dir, self.blob_dir, self.cache_dir, self.dataset_cache_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


def project_root() -> Path:
    """仓库根：优先 ONYX_ROOT，否则向上找含 pyproject.toml 的目录。"""
    if env := os.environ.get("ONYX_ROOT"):
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here.parent


def load_settings(data_dir: str | os.PathLike[str] | None = None) -> Settings:
    root = project_root()
    chosen = data_dir or os.environ.get("ONYX_DATA_DIR") or (root / DEFAULT_DATA_DIR_NAME)
    return Settings(data_dir=Path(chosen).resolve())
