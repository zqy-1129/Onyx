"""共享 fixture。

除通用夹具外还提供扩展点测试需要的两样东西：
`plugins_example/` 的路径、以及按 group 伪造 entry points 的 `entry_points_env`。
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore, MemoryBlobStore

# `plugins_example/` 里的两个包是**外部插件样板**，不是 onyx 的依赖。
# 把它们挂进 sys.path，测试就能用伪造的 entry point 指向真实源码，
# 而不需要真的执行安装步骤（CI 里装包会把扩展点测试变成"看运气"的检查）。
_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("example_task", "example_provider"):
    _path = str(_ROOT / "plugins_example" / _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(autouse=True)
def _clean_discovery():
    """发现缓存与失败台账是进程级的：测试之间必须归零。

    否则一个测试里登记的"坏插件"会让后面所有断言看到凭空的失败，
    而这种串扰通常表现为"单独跑通过、全量跑失败"。
    """
    from onyx.discovery import clear_cache, reset_failures

    clear_cache()
    reset_failures()
    yield
    clear_cache()
    reset_failures()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def blobs(tmp_path) -> FileBlobStore:
    return FileBlobStore(tmp_path / "blobs")


@pytest.fixture
def mem_blobs() -> MemoryBlobStore:
    return MemoryBlobStore()


class EntryPointsEnv:
    """按 group 伪造 entry points 声明（等价于"这个包已经装好了"）。

    `install(group, "name=module:attr", ...)`；未声明的 group 返回空，
    所以真实环境里恰好装着的其它插件不会干扰断言。
    """

    def __init__(self, table: dict[str, tuple[EntryPoint, ...]]) -> None:
        self.table = table

    def install(self, group: str, *entries: str) -> None:
        import onyx.discovery as discovery

        parsed: list[EntryPoint] = []
        for text in entries:
            name, _, value = text.partition("=")
            parsed.append(EntryPoint(name.strip(), value.strip(), group))
        self.table[group] = tuple(parsed)
        discovery.clear_cache()

    def fake(self, *, group: str | None = None) -> tuple[EntryPoint, ...]:
        if group is None:
            return tuple(e for items in self.table.values() for e in items)
        return self.table.get(group, ())


@pytest.fixture
def entry_points_env(monkeypatch) -> EntryPointsEnv:
    import onyx.discovery as discovery

    env = EntryPointsEnv({})
    monkeypatch.setattr(discovery, "entry_points", env.fake)
    return env
