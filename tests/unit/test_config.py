"""S20 部署配置 `onyx.toml`：优先级、写了不生效的检出、以及每条命令真的读到了它。

配置系统最坏的失败不是报错，而是"文件躺在那里、命令照样用默认值"。
所以这里的断言集中在三件事：
1. 优先级是**一条规则**（flag > 环境 > 文件 > 默认），而不是每条命令各自的习惯；
2. 文件里内核不认识的键与类型不对的键会被**记下来并指名**，不会静默回落；
3. 每一项配置都能被证明走到了真正的消费者那里（provider、锁、窗口、sandbox、serve）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from onyx.cli import _policy_from_cli, _resolved_model, _runtime, app
from onyx.config import ConfigError, config_path, effective, load_config, pick
from onyx.settings import load_settings

runner = CliRunner()


def _write(tmp_path, monkeypatch, text: str) -> Path:
    path = tmp_path / "onyx.toml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("ONYX_CONFIG", str(path))
    return path


# ── 优先级 ─────────────────────────────────────────────────────────
def test_pick_prefers_explicit_and_keeps_falsy_values():
    assert pick("--flag", "file", "default") == "--flag"
    assert pick(None, "file", "default") == "file"
    assert pick(None, None, "default") == "default"
    # 0 和空串是"给了"，不是"没给"：把它们当未设就等于吞掉用户的显式输入
    assert pick(0, 5, 9) == 0
    assert pick("", "x", "d") == ""


def test_discovery_order_env_beats_root_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_ROOT", str(tmp_path))
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "r"\n', encoding="utf-8")
    root_file = tmp_path / "onyx.toml"
    root_file.write_text('[serve]\nport = 9999\n', encoding="utf-8")

    assert config_path() == root_file
    elsewhere = _write(tmp_path, monkeypatch, '[serve]\nport = 7777\n')
    assert config_path() == elsewhere, "ONYX_CONFIG 应该压过仓库根那份"


def test_data_dir_precedence_flag_env_file(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, f'data_dir = "{(tmp_path / "from-file").as_posix()}"')
    assert load_settings().data_dir.name == "from-file"

    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path / "from-env"))
    assert load_settings().data_dir.name == "from-env", "环境变量优先于文件"
    assert load_settings("d/explicit").data_dir.name == "explicit"


# ── 坏配置必须出声 ─────────────────────────────────────────────────
def test_explicit_missing_file_is_an_error(tmp_path, monkeypatch):
    """显式指定的路径不存在时不能"当作没配置"继续跑。

    继续跑会用默认数据目录/默认引擎地址做真实写入，而人以为自己的配置生效了。
    """
    missing = tmp_path / "nope.toml"
    monkeypatch.setenv("ONYX_CONFIG", str(missing))

    with pytest.raises(ConfigError, match="不存在"):
        load_config()


def test_syntax_error_names_the_file(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "[serve]\nport = ")

    with pytest.raises(ConfigError, match="语法错"):
        load_config()


def test_unknown_keys_and_bad_types_are_collected(tmp_path, monkeypatch):
    path = _write(tmp_path, monkeypatch, (
        'stray = 1\n'
        '[serve]\npot = 1234\nport = "8787"\n'
        '[colour]\nhue = "red"\n'
    ))

    cfg = load_config(path)

    assert "serve.pot" in cfg.unknown
    assert "[colour]" in cfg.unknown
    assert "stray" in cfg.unknown
    assert any("serve.port" in problem for problem in cfg.problems), "类型不对要指名"
    assert cfg.serve.port is None, "类型不对的键不冒充生效值"


def test_bool_is_not_an_integer_port(tmp_path, monkeypatch):
    """`port = true` 在 Python 里就是 1 号端口。这类手滑必须被拒，而不是猜个意思。"""
    cfg = load_config(_write(tmp_path, monkeypatch, "[serve]\nport = true\n"))

    assert cfg.serve.port is None
    assert any("不能是布尔值" in problem for problem in cfg.problems)


def test_empty_string_in_file_means_unset(tmp_path, monkeypatch):
    cfg = load_config(_write(tmp_path, monkeypatch, '[provider]\nbase_url = ""\n'))

    assert cfg.provider.base_url is None, "文件里的空串通常是「这行我没填」"


# ── 每一项配置都要被真的消费 ───────────────────────────────────────
def test_provider_base_url_reaches_the_gateway(tmp_path, monkeypatch):
    """配置文件里换引擎地址，必须真的传到 build_provider，而不是只有 --url 管用。"""
    seen: dict[str, object] = {}

    def fake_build_runtime(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("sentinel")

    import onyx.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "build_runtime", fake_build_runtime)
    _write(tmp_path, monkeypatch, '[provider]\nkind = "openai-compat"\nbase_url = "http://127.0.0.1:8001/v1"\n')

    with pytest.raises(RuntimeError, match="sentinel"):
        _runtime(None, None)

    assert seen["base_url"] == "http://127.0.0.1:8001/v1"
    assert seen["provider_kind"] == "openai-compat"
    # kind 换了，落库的 provider_id 也必须跟着换：否则"这个数字来自哪个引擎"是假的
    assert seen["provider_id"] == "openai-compat-local"
    assert seen["sample_gpu"] is False, "非 ollama 通道不去猜显存"

    with pytest.raises(RuntimeError, match="sentinel"):
        _runtime("http://127.0.0.1:9999", None)
    assert seen["base_url"] == "http://127.0.0.1:9999", "显式 flag 必须赢过配置文件"


def test_sink_defaults_come_from_config(tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    def fake_build_runtime(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("sentinel")

    import onyx.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "build_runtime", fake_build_runtime)
    _write(tmp_path, monkeypatch, '[sinks]\nevents = ["jsonl"]\n')

    with pytest.raises(RuntimeError, match="sentinel"):
        _runtime(None, None)
    assert seen["event_sinks"] == ("jsonl",)

    with pytest.raises(RuntimeError, match="sentinel"):
        _runtime(None, None, event_sinks=("null",))
    assert seen["event_sinks"] == ("null",), "命令行点名了就不要往里面塞配置的默认"


def test_sandbox_policy_comes_from_config(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, '[sandbox]\nallowed_side_effects = ["read", "network"]\n')

    policy = _policy_from_cli(None)

    assert {str(effect) for effect in policy.allowed_side_effects} >= {"network"}
    # 常驻放开的同时不能悄悄放大动态导入的白名单
    assert policy.allowed_impl_prefixes == ("onyx.tools.builtin.", "mcp:")


def test_sandbox_bad_side_effect_name_names_the_source(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, '[sandbox]\nallowed_side_effects = ["bogus"]\n')

    result = runner.invoke(app, ["tools", "run", "echo"])

    assert result.exit_code == 2
    assert "bogus" in result.output
    assert "allowed_side_effects" in result.output, "要说清这个值是从配置文件来的还是 --allow 给的"


def test_rotation_windows_come_from_config(tmp_path, monkeypatch):
    from onyx.store.db import Database

    _write(tmp_path, monkeypatch, '[retention]\nraw_after = "7d"\ntrace_after = "14d"\n')
    data_dir = tmp_path / "data"
    monkeypatch.setenv("ONYX_DATA_DIR", str(data_dir))
    Database((data_dir / "onyx.sqlite").resolve()).close()
    calls: dict[str, object] = {}

    import onyx.store.retention as retention_module

    real_sweep = retention_module.sweep

    def spy(db, store, **kwargs):
        calls.update(kwargs)
        return real_sweep(db, store, **kwargs)

    monkeypatch.setattr(retention_module, "sweep", spy)
    monkeypatch.setattr("onyx.cli.sweep", spy)

    result = runner.invoke(app, ["rotate", "--json"])

    assert result.exit_code == 0, result.output
    assert calls["raw_after"] == "7d" and calls["trace_after"] == "14d"


def test_model_default_from_config_and_stop_when_absent(tmp_path, monkeypatch):
    import typer as typer_module

    _write(tmp_path, monkeypatch, '[provider]\nmodel = "qwen3.5:9b"\n')
    assert _resolved_model(None, "onyx chat") == "qwen3.5:9b"
    assert _resolved_model("other", "onyx chat") == "other"

    monkeypatch.delenv("ONYX_CONFIG")
    with pytest.raises(typer_module.Exit):
        _resolved_model(None, "onyx chat")


def test_chat_stops_when_no_model_is_known(tmp_path, monkeypatch):
    """猜一个模型名去打通引擎，比停下来问更糟：分数与 trace 都会来自错的机器。"""
    result = runner.invoke(app, ["chat", "你好"])

    assert result.exit_code == 2
    assert "--model" in result.output and "[provider].model" in result.output


def test_serve_reads_host_port_lock_and_stale(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    served: dict[str, object] = {}

    def fake_create_app(**kwargs):
        captured.update(kwargs)
        return object()

    import uvicorn

    import onyx.api.app as app_module

    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(kw))
    # 0.0.0.0 现在必须带 token（S21 的姿态）：这里正是要验证"配置文件里的 host 真的传到了 uvicorn"
    _write(tmp_path, monkeypatch, (
        '[serve]\nhost = "0.0.0.0"\nport = 9123\ntoken = "cfg-token"\n'
        '[gpu]\nlock_path = "D:/tmp/onyx-gpu-2.lock"\nstale_after_s = 333.0\n'
    ))

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    assert captured["gpu_lock_path"] == "D:/tmp/onyx-gpu-2.lock"
    assert captured["gpu_stale_after_s"] == 333.0
    assert captured["token"] == "cfg-token"
    assert served["host"] == "0.0.0.0" and served["port"] == 9123

    with_explicit = runner.invoke(app, ["serve", "--gpu-lock", "D:/tmp/flag.lock"])
    assert with_explicit.exit_code == 0, with_explicit.output
    assert captured["gpu_lock_path"] == Path("D:/tmp/flag.lock"), "flag 必须赢过文件"


# ── 可观测性：`config show` 与 doctor ──────────────────────────────
def test_broken_config_stops_ordinary_commands(tmp_path, monkeypatch):
    """回归测试：坏配置曾被**静默忽略**。

    当时 `--config` 指到一个不存在的文件，只要 ONYX_DATA_DIR 设过，`db info` 照常跑——
    因为每个值都能从更高优先级的来源拿到，没有任何一条路径会去读那份文件。
    坏配置被忽略比没有配置更糟：人以为 `[serve].port` 已经生效了。
    """
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))

    result = runner.invoke(app, ["--config", str(tmp_path / "nope.toml"), "db", "info"])

    assert result.exit_code == 2, result.output
    assert "配置文件" in (result.output + getattr(result, "stderr", ""))


def test_doctor_survives_a_broken_config_because_it_is_the_diagnosis(tmp_path, monkeypatch):
    """体检是"查这份文件怎么了"的入口，所以它必须能在文件坏掉时照样跑并指名问题。"""
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    bad = tmp_path / "bad.toml"
    bad.write_text("[serve]\nport = \n", encoding="utf-8")

    result = runner.invoke(app, ["--config", str(bad), "doctor", "--skip-network"])

    assert result.exit_code == 1, result.output
    assert "配置文件" in result.output and "语法错" in result.output


def test_effective_names_the_env_layer(tmp_path, monkeypatch):
    """三层里"环境"必须是独立一档，否则报告会把它说成默认值。"""
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path / "from-env"))

    rows = {row.key: row for row in effective(load_config())}

    assert rows["data_dir"].source == "env"
    assert rows["data_dir"].value == str(tmp_path / "from-env")


def test_project_root_falls_back_to_the_repository(tmp_path, monkeypatch):
    """没有 ONYX_ROOT 时，根目录靠向上找 `pyproject.toml`——这条路径必须在真仓库上成立。

    它同时是"配置文件默认在哪"与"默认数据目录在哪"的共同前提，所以值得钉住。
    """
    from onyx.config import project_root

    monkeypatch.delenv("ONYX_ROOT", raising=False)
    monkeypatch.delenv("ONYX_DATA_DIR", raising=False)
    # 指一份空配置：否则开发者自己那份 `<根>/onyx.toml` 里的 data_dir 会让这条断言假红
    _write(tmp_path, monkeypatch, "")

    root = project_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "onyx").is_dir(), "找到的是仓库根，不是某个临时目录"
    assert load_settings().data_dir == (root / ".data").resolve()


def test_section_that_is_not_a_table_is_reported(tmp_path, monkeypatch):
    """`serve = 1` 这种写法不能崩，也不能被当成"没配"。"""
    cfg = load_config(_write(tmp_path, monkeypatch, "serve = 1\n"))

    assert any("serve" in problem for problem in cfg.problems)
    assert cfg.serve.port is None


def test_effective_reports_which_layer_won(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, '[serve]\nport = 9000\n')

    rows = {row.key: row for row in effective(load_config())}

    assert rows["serve.port"].value == 9000 and rows["serve.port"].source == "file"
    assert rows["serve.host"].source == "default"
    assert rows["provider.base_url"].value == "http://127.0.0.1:11434"


def test_config_show_lists_every_key_and_flags_unknown(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "[serve]\nport = 9000\npot = 1\n")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0, result.output
    assert "serve.port" in result.output and "[文件]" in result.output.replace("［", "[")
    assert "9000" in result.output
    assert "serve.pot" in result.output, "写了不生效的键必须在输出里可见"


def test_config_show_json(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, '[provider]\nmodel = "m1"\n')

    result = runner.invoke(app, ["config", "show", "--json"])

    body = json.loads(result.output)
    assert body["loaded"] is True
    assert body["effective"]["provider.model"] == "m1"
    assert body["source"]["provider.model"] == "file"
    assert body["problems"] == []


def test_doctor_reports_config_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "240")

    no_file = runner.invoke(app, ["doctor", "--skip-network"])
    assert "没有配置文件" in no_file.output, no_file.output

    _write(tmp_path, monkeypatch, "[serve]\npot = 1\n")
    bad = runner.invoke(app, ["doctor", "--skip-network"])
    assert bad.exit_code == 1, bad.output
    assert "serve.pot" in bad.output

    monkeypatch.setenv("ONYX_CONFIG", str(tmp_path / "gone.toml"))
    unreadable = runner.invoke(app, ["doctor", "--skip-network"])
    assert unreadable.exit_code == 1, unreadable.output
    assert "配置文件" in unreadable.output, "配置文件读不出来时 doctor 仍然要能跑并指名它"
