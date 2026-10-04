"""运行期路径与配置解析。

数据目录的优先级只有一条：显式参数 > `ONYX_DATA_DIR` > 配置文件 `data_dir` > `<根>/.data`。
`project_root()` 住在 `onyx.config`（它同时用来找配置文件），这里只做再导出，
免得"找根目录"这件事有两份实现。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from onyx.config import ENV_DATA_DIR, load_config, project_root

DEFAULT_DATA_DIR_NAME = ".data"

__all__ = ["DEFAULT_DATA_DIR_NAME", "Settings", "load_settings", "project_root"]


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


def load_settings(data_dir: str | os.PathLike[str] | None = None) -> Settings:
    """解析数据目录。空串不算"设过了"——它通常是脚本变量没取到。"""
    chosen = data_dir or os.environ.get(ENV_DATA_DIR) or load_config().data_dir
    root = project_root()
    return Settings(data_dir=Path(chosen or root / DEFAULT_DATA_DIR_NAME).resolve())
