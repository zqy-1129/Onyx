"""共享 fixture。"""

from __future__ import annotations

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore, MemoryBlobStore


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def blobs(tmp_path) -> FileBlobStore:
    return FileBlobStore(tmp_path / "blobs")


@pytest.fixture
def mem_blobs() -> MemoryBlobStore:
    return MemoryBlobStore()
