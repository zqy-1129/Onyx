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


def test_delete_returns_the_bytes_it_freed(store):
    """`delete` 的返回值是保留策略的记账依据：字节数，不是"成功/失败"。"""
    payload = "原始响应 " * 20
    ref = store.put(payload, media="application/json")
    size = store.stat(ref).size

    assert store.delete(ref) == size
    assert not store.exists(ref)
    assert store.delete(ref) == 0, "本来就不存在 ⇒ 0 字节，不是报错也不是负数"
    assert ref not in store.iter_refs()
    assert store.total_bytes() == 0


def test_delete_rejects_an_illegal_ref_without_touching_anything(store):
    """非法 ref 在拼路径之前就 must 炸——否则"删文件"这件事本身可以是穿越向量。"""
    kept = store.put("证据")
    with pytest.raises(InvalidBlobRef):
        store.delete("sha256:" + "../" * 8 + "etc/passwd")
    assert store.exists(kept)


def test_sidecar_is_removed_with_the_blob(tmp_path):
    """media 侧车跟着 blob 走：留下它，`total_bytes` 之外还会攒一堆指向空气的元数据。"""
    store = FileBlobStore(tmp_path / "b")
    ref = store.put("html", media="text/html")
    sidecar = store.stat(ref).path.with_suffix(".meta.json")
    assert sidecar.exists()

    assert store.delete(ref) == 4
    assert not sidecar.exists()
    assert list((tmp_path / "b").rglob("*.meta.json")) == []


def test_iter_refs_ignores_stray_files(tmp_path):
    """目录里被人塞一个 README 不该变成"一个可回收的 blob"。"""
    store = FileBlobStore(tmp_path / "b")
    ref = store.put("real")
    (tmp_path / "b" / "README.md").write_text("有人放的", encoding="utf-8")

    assert store.iter_refs() == [ref]
