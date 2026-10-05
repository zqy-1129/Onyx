"""S7 承诺、S19 补齐的两项体检：磁盘余量、token 计量档位。

体检的全部价值在于"红的那一项说得出下一步做什么"，以及"没红的时候它说的是真话"。
所以这里既测失败路径，也测一条诚实声明：`hf_tokenizer` / `gguf_vocab` 两档在本版本
没有实现，体检必须把这件事写在明面上——否则看板会显得比实际更能精确复算。
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from onyx.cli import (
    FITTED_MIN_SAMPLES,
    MIN_FREE_BYTES,
    _check_disk,
    _check_token_tiers,
    _dir_bytes,
    app,
)
from onyx.core.types import Generation, GenerationRequest, Message
from onyx.llm.measurement.fidelity import CounterContext, FittedCounter
from onyx.settings import load_settings
from onyx.store.db import Database
from onyx.store.records import ModelRecord, ProviderRecord
from onyx.store.repos import ModelRepo

runner = CliRunner()


def _request() -> GenerationRequest:
    return GenerationRequest(model="m", messages=(Message(role="user", content="你好"),))


def _generation() -> Generation:
    return Generation(text="回答", finish_reason="stop")


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "t.sqlite") as database:
        yield database


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "240")
    return load_settings().ensure_dirs()


def _model(repo: ModelRepo, name: str, *, ratio=None, n: int = 0) -> str:
    repo.upsert_provider(ProviderRecord(
        id="ollama-local", kind="ollama", base_url="http://127.0.0.1:11434", api_style="native"
    ))
    model_id = repo.upsert_model(ModelRecord(
        id=f"ollama-local/{name}", provider_id="ollama-local", name=name,
        remote_model=name,
    ))
    if ratio is not None or n:
        repo.update_model(model_id, usage_ratio=ratio, usage_ratio_n=n)
    return model_id


# ── 磁盘余量 ───────────────────────────────────────────────────────
def test_disk_check_fails_below_the_floor(tmp_path, monkeypatch):
    """余量不足 ⇒ 红，而且修法要具体到"先回收还是先挪目录"。"""
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda _p: SimpleNamespace(total=100 * 1024 ** 3, used=99 * 1024 ** 3,
                                   free=MIN_FREE_BYTES - 1),
    )
    check = _check_disk(load_settings().ensure_dirs())

    assert check.ok is False
    assert "rotate --apply" in check.hint
    assert "ONYX_DATA_DIR" in check.hint


def test_disk_check_passes_with_room_and_shows_the_footprint(tmp_path, monkeypatch, data_dir):
    (data_dir.blob_dir / "x").write_text("0" * 1234, encoding="utf-8")
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda _p: SimpleNamespace(total=100 * 1024 ** 3, used=50 * 1024 ** 3, free=50 * 1024 ** 3),
    )

    check = _check_disk(data_dir)

    assert check.ok is True
    assert _dir_bytes(data_dir.data_dir) >= 1234
    assert "剩余" in check.detail and ".data 现占" in check.detail


def test_disk_check_reports_when_it_cannot_ask(tmp_path, monkeypatch):
    """问不出来要说"问不出来"，不能装作余量充足。"""
    def boom(_path):
        raise OSError("volume gone")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    check = _check_disk(load_settings().ensure_dirs())

    assert check.ok is False
    assert "问不出来" in check.detail


# ── token 计量档位 ─────────────────────────────────────────────────
def test_tier_check_names_the_models_that_cannot_be_attributed(db):
    repo = ModelRepo(db)
    _model(repo, "calibrated", ratio=0.62, n=FITTED_MIN_SAMPLES)
    _model(repo, "fresh")

    check = _check_token_tiers(db)

    assert check.ok is False
    assert "fresh" in check.detail, "要指名是哪个模型只能退回启发式"
    assert "calibrate" in check.hint


def test_tier_check_boundary_is_the_same_number_the_counter_uses(db):
    """体检与计数必须共用一个门槛。

    分成两个数时会出现最糟的组合：体检绿着，而实际计数早已退回 heuristic/low。
    """
    repo = ModelRepo(db)
    _model(repo, "just-short", ratio=0.62, n=FITTED_MIN_SAMPLES - 1)

    check = _check_token_tiers(db)
    sample = FittedCounter().count(
        _request(), _generation(),
        CounterContext(fitted_ratio=0.62, fitted_n=FITTED_MIN_SAMPLES - 1),
    )

    assert check.ok is False
    assert sample is not None and sample.ok is False
    assert str(FITTED_MIN_SAMPLES) in (sample.note or "")


def test_tier_check_admits_two_tiers_are_not_implemented(db):
    """能力声明要按"这台机器上真能产出什么"来说，而不是按设计稿上有几档。"""
    check = _check_token_tiers(db)

    assert check.ok is True, "库里没模型不是故障"
    assert "hf_tokenizer" in check.detail and "未实现" in check.detail
    assert "models sync" in check.hint


def test_tier_check_passes_when_every_model_is_calibrated(db):
    repo = ModelRepo(db)
    _model(repo, "a", ratio=0.7, n=FITTED_MIN_SAMPLES + 20)

    check = _check_token_tiers(db)

    assert check.ok is True
    assert "未标定 0" in check.detail


# ── CLI 出口 ───────────────────────────────────────────────────────
def test_doctor_lists_both_new_checks(data_dir, monkeypatch):
    monkeypatch.setattr("onyx.cli.MIN_FREE_BYTES", 1)
    Database(data_dir.db_path).close()

    result = runner.invoke(app, ["doctor", "--skip-network"])

    assert result.exit_code == 0, result.output
    for item in ("磁盘余量", "token 计量档位", "blob 引用完整", "扩展点插件全部可加载", "告警"):
        assert item in result.output


# ── S29：告警这项体检 ──────────────────────────────────────────────
def _alerts_check(tmp_path, monkeypatch, body: str, db):
    """用真实配置文件喂进体检（不是构造 Config 对象）：装配口径也要被测到。"""
    from onyx.cli import _check_alerts
    from onyx.config import load_config
    from onyx.settings import load_settings

    monkeypatch.delenv("ONYX_ALERT_WEBHOOK_URL", raising=False)
    cfg_path = tmp_path / "onyx.toml"
    cfg_path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    settings = load_settings()
    return _check_alerts(load_config(cfg_path), settings, db)


def test_alert_check_is_green_and_reads_out_its_posture(tmp_path, monkeypatch, db):
    from onyx.store.records import AlertTriggerRecord
    from onyx.store.repos import AlertRepo

    AlertRepo(db).insert_trigger(AlertTriggerRecord(
        id="t1", created_at="2026-10-05T03:00:00+00:00", code="CONTEXT_OVERFLOW",
        severity="error", rule={}, n_in_window=1, window_s=300, channel="file",
        status="sent", detail="写入 1 行"))
    check = _alerts_check(tmp_path, monkeypatch, "[alerts]\nwindow_s = 120\n", db)
    assert check.ok is True
    assert "120s 内满 1 次" in check.detail and "出口 file" in check.detail
    # 投递留痕要能在体检里看见，否则"通知有没有发过"还要再开一条命令
    assert "今天投递 1 次" in check.detail


def test_alert_check_says_disabled_instead_of_looking_broken(tmp_path, monkeypatch, db):
    check = _alerts_check(tmp_path, monkeypatch, "[alerts]\nenabled = false\n", db)
    assert check.ok is True and "未启用" in check.detail
    assert "serve" in check.hint, "要说清通知由谁发，否则人会在 CLI 进程等通知"


def test_alert_check_goes_red_when_every_recent_delivery_failed(tmp_path, monkeypatch, db):
    from onyx.store.records import AlertTriggerRecord
    from onyx.store.repos import AlertRepo

    repo = AlertRepo(db)
    for i in range(5):
        repo.insert_trigger(AlertTriggerRecord(
            id=f"f{i}", created_at="2026-10-05T03:00:00+00:00", code="CONTEXT_OVERFLOW",
            severity="error", rule={}, n_in_window=1, window_s=300, channel="webhook",
            status="failed", detail="ConnectError: 拒绝连接"))
    check = _alerts_check(tmp_path, monkeypatch, "[alerts]\ncooldown_s = 0\n", db)
    assert check.ok is False
    assert "全失败" in check.hint and "alerts test" in check.hint
    assert "ConnectError" in check.hint, "红的那一项要带着原因，否则又要人去查"


def test_alert_check_partial_failure_is_only_a_note(tmp_path, monkeypatch, db):
    """一次偶发失败（接收端重启）不该把体检染红，但必须在 detail 里看得见。"""
    from onyx.store.records import AlertTriggerRecord
    from onyx.store.repos import AlertRepo

    repo = AlertRepo(db)
    repo.insert_trigger(AlertTriggerRecord(
        id="f1", created_at="2026-10-05T03:00:00+00:00", code="X", severity="error",
        rule={}, n_in_window=1, window_s=300, channel="webhook", status="failed", detail="超时"))
    repo.insert_trigger(AlertTriggerRecord(
        id="s1", created_at="2026-10-05T03:10:00+00:00", code="X", severity="error",
        rule={}, n_in_window=1, window_s=300, channel="file", status="sent", detail="写入"))
    check = _alerts_check(tmp_path, monkeypatch, "[alerts]\n", db)
    assert check.ok is True and "最近 2 次里 1 次失败" in check.detail


def test_alert_check_goes_red_when_the_file_channel_cannot_write(tmp_path, monkeypatch, db):
    """文件出口写不下去等于没有出口：这必须是红项，而不是"配置看起来正常"。"""
    blocker = tmp_path / "blocked"
    blocker.write_text("我是个文件，不是目录", encoding="utf-8")
    body = f'[alerts]\nfile = "{(blocker / "alerts.jsonl").as_posix()}"\n'
    check = _alerts_check(tmp_path, monkeypatch, body, db)
    assert check.ok is False and "写不下去" in check.detail


def test_doctor_exits_nonzero_when_a_check_fails(data_dir, monkeypatch):
    monkeypatch.setattr("onyx.cli.MIN_FREE_BYTES", 10 ** 18)
    Database(data_dir.db_path).close()

    result = runner.invoke(app, ["doctor", "--skip-network"])

    assert result.exit_code == 1
    assert "磁盘余量" in result.output
