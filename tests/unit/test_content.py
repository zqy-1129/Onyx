from __future__ import annotations

import pytest

from onyx.core.content import FileBlobStore, MemoryBlobStore
from onyx.core.errors import BlobNotFound, InvalidBlobRef

STORES = [FileBlobStore, MemoryBlobStore]


@pytest.fixture(params=STORES, ids=["file", "memory"])
def store(request, tmp_path):
    return request.param(tmp_path / "b") if request.param is FileBlobStore else request.param()


def test_put_get_roundtrip(store):
    ref = store.put("你好, Onyx")
    assert ref.startswith("sha256:") and len(ref) == 71
    assert store.get_text(ref) == "你好, Onyx"


def test_same_content_same_ref_dedupe(store):
    a = store.put("duplicate payload")
    b = store.put("duplicate payload")
    assert a == b


def test_json_roundtrip(store):
    ref = store.put_json({"messages": [{"role": "user", "content": "hi"}], "n": 1})
    assert store.get_json(ref) == {"messages": [{"role": "user", "content": "hi"}], "n": 1}


def test_media_type_preserved(store):
    ref = store.put("<html/>", media="text/html")
    assert store.stat(ref).media == "text/html"
    assert store.stat(ref).size == 7


def test_missing_blob_raises(store):
    ref = store.put("x")
    other = "sha256:" + "0" * 64
    assert store.exists(ref)
    assert not store.exists(other)
    with pytest.raises(BlobNotFound):
        store.get(other)


@pytest.mark.parametrize(
    "bad_ref",
    [
        "",
        "sha256:deadbeef",
        "md5:" + "a" * 32,
        "sha256:" + "../" * 10 + "etc/passwd",
        "sha256:" + "A" * 64,
        "../../onyx.sqlite",
        "sha256:" + "0" * 63,
    ],
)
def test_invalid_refs_rejected(store, bad_ref):
    """安全断言：ref 参与路径拼接前必须严格校验，杜绝目录穿越。"""
    with pytest.raises(InvalidBlobRef):
        store.get(bad_ref)
    assert store.exists(bad_ref) is False


def test_atomic_write_leaves_no_tmp(tmp_path):
    store = FileBlobStore(tmp_path / "b")
    store.put("payload")
    assert not list((tmp_path / "b").rglob("*.tmp"))


def test_sharded_layout(tmp_path):
    store = FileBlobStore(tmp_path / "b")
    ref = store.put("payload")
    hexdig = ref.split(":")[1]
    assert store.stat(ref).path == tmp_path / "b" / hexdig[:2] / hexdig[2:4] / hexdig


def test_file_store_maintenance(tmp_path):
    store = FileBlobStore(tmp_path / "b")
    refs = {store.put(f"blob-{i}") for i in range(5)}
    assert set(store.iter_refs()) == refs
    assert store.total_bytes() == sum(len(f"blob-{i}".encode()) for i in range(5))
