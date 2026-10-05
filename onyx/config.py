"""部署配置：一个 `onyx.toml` 说清"这台机器上 Onyx 怎么跑"。

为什么是 TOML 而不是 DESIGN §13 里写的 `onyx.yaml`：本模块被 `onyx.settings` 导入，
而 `settings` 在 `runtime` / `store` 这条**核心管道**上——用 YAML 就等于让只用库不用 CLI 的
人也得装 `runtime` extra（PyYAML 在那里，工具定义导入确实需要它）。
`tomllib` 是标准库，配置层因此零额外依赖；格式也与 `pyproject.toml` 一致。
承诺（配置外置、优先级一条规则）没变。
格式换了，承诺没换：散在各条命令 flag 里的部署取舍收到一处。

**优先级只有一条规则**：显式 flag > 环境变量 > 配置文件 > 内建默认。
实现方式很关键——flag 的内建默认必须一律是 `None`。只要 flag 带着具体默认值
（比如 `--url http://127.0.0.1:11434`），它就永远赢过配置文件，配置文件当场变成摆设，
而且没有任何地方会报错。这是配置系统最常见的死法，所以 `doctor` 有一项专门盯它。

另一个死法是"写了不生效"：内核不消费的键被静默忽略。因此这里带一份 schema，
文件里出现 schema 之外的键会被记成 `unknown` 由 `doctor` 指名；
反过来，schema 里的每一个键都必须有真实消费者，加键之前先找到那行代码。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_NAME = "onyx.toml"
ENV_CONFIG_PATH = "ONYX_CONFIG"
ENV_DATA_DIR = "ONYX_DATA_DIR"
#: token 优先走环境变量而不是配置文件：写进文件的 token 会跟着备份、截图和 git status 漂走。
ENV_TOKEN = "ONYX_SERVE_TOKEN"


class ConfigError(ValueError):
    """配置文件读不出来（语法错、显式给的路径不存在）。

    不能"忽略坏文件继续跑"：那会让人以为锁路径/数据目录已经改了，而它其实还是默认值。
    """


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    kind: str | None = None
    base_url: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class GpuConfig:
    lock_path: str | None = None
    #: 心跳超过多少秒没更新就认为持有者死了。必须大于单条样本的最长耗时。
    stale_after_s: float | None = None


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    raw_after: str | None = None
    trace_after: str | None = None


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    allowed_side_effects: tuple[str, ...] | None = None
    allowed_impl_prefixes: tuple[str, ...] | None = None
    default_timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class ServeConfig:
    host: str | None = None
    port: int | None = None
    #: 局域网共享看板的闸门。优先用环境变量 `ONYX_SERVE_TOKEN`：
    #: 把 token 写进文件意味着它会跟着备份、截图和 git 状态一起漂走。
    token: str | None = None
    #: 只读 = 挡住"别人往你的 GPU 上打请求 / unload 你的模型"，挡不住读。
    read_only: bool | None = None


@dataclass(frozen=True, slots=True)
class AlertConfig:
    """`[alerts]`：什么异常、到什么程度、发到哪。

    判据的默认值不在这里抄一份——它们的家是 `obs/alerts/rules.py`，
    这里只留"人改过的值"，没改的就是 None（沿用 S20 的分层规矩）。
    """

    enabled: bool | None = None
    severities: tuple[str, ...] | None = None
    codes: tuple[str, ...] | None = None
    exclude_codes: tuple[str, ...] | None = None
    window_s: int | None = None
    min_count: int | None = None
    cooldown_s: int | None = None
    poll_s: float | None = None
    #: 本地文件出口的路径。没写就是 `<数据目录>/alerts/alerts.jsonl`（按天分片）
    file: str | None = None


@dataclass(frozen=True, slots=True)
class Config:
    data_dir: str | None = None
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    sinks: tuple[str, ...] | None = None
    #: 真正读到的文件（没有就是空元组）——报告里必须说清数是从哪份文件来的
    sources: tuple[Path, ...] = ()
    #: schema 之外的键（写了不生效 ⇒ doctor 指名）
    unknown: tuple[str, ...] = ()
    #: 类型不对的键。不阻断运行，但也不能装作它生效了
    problems: tuple[str, ...] = ()

    @property
    def loaded(self) -> bool:
        return bool(self.sources)


#: schema 是唯一事实来源：这里的每个键都必须在某处被真的消费掉。
#: (顶层 None 表示"顶层键") → {键: 允许的类型}
_SCHEMA: dict[str | None, dict[str, tuple[type, ...]]] = {
    None: {"data_dir": (str,)},
    "provider": {"kind": (str,), "base_url": (str,), "model": (str,)},
    "gpu": {"lock_path": (str,), "stale_after_s": (int, float)},
    "retention": {"raw_after": (str,), "trace_after": (str,)},
    "sandbox": {
        "allowed_side_effects": (list,),
        "allowed_impl_prefixes": (list,),
        "default_timeout_ms": (int,),
    },
    "serve": {"host": (str,), "port": (int,), "token": (str,), "read_only": (bool,)},
    "alerts": {
        "enabled": (bool,),
        "severities": (list,), "codes": (list,), "exclude_codes": (list,),
        "window_s": (int,), "min_count": (int,), "cooldown_s": (int,), "poll_s": (int, float),
        "file": (str,),
    },
    "sinks": {"events": (list,)},
}


def project_root() -> Path:
    """仓库/安装根：优先 ONYX_ROOT，否则向上找含 pyproject.toml 的目录。"""
    if env := os.environ.get("ONYX_ROOT"):
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here.parent


def config_path(explicit: Path | str | None = None) -> Path | None:
    """`--config` > `ONYX_CONFIG` > `<根>/onyx.toml`（存在才返回）。"""
    if explicit:
        return Path(explicit)
    if env := os.environ.get(ENV_CONFIG_PATH):
        return Path(env)
    guess = project_root() / CONFIG_NAME
    return guess if guess.exists() else None


def pick[T](*values: T | None) -> T | None:
    """按顺序取第一个"给过了"的值——`None` 才是没给，空字符串和 0 都是给了。

    用 `or` 写这个函数会把 `--port 0`、`--model ""` 这类显式值吃成"没给"，
    然后悄悄回落到默认值：用户看到的是默认值的行为，却以为自己设过。
    """
    for value in values:
        if value is not None:
            return value
    return None


def _check_type(section: str | None, key: str, value: Any, allowed: tuple[type, ...]) -> str | None:
    """返回 None 表示通过；否则返回一条人能照着改的问题描述。"""
    label = f"{section}.{key}" if section else key
    if isinstance(value, bool) and bool not in allowed:
        # bool 是 int 的子类：`port = true` 会被当成 1 号端口，那是凭手滑造成的假配置
        return f"{label} 必须是 {'/'.join(t.__name__ for t in allowed)}，不能是布尔值"
    if not isinstance(value, allowed):
        return f"{label} 必须是 {'/'.join(t.__name__ for t in allowed)}，实际是 {type(value).__name__}"
    return None


def _str_list(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    return tuple(str(item) for item in value)


def _text(value: Any) -> str | None:
    """配置文件里写空串通常表示"这行我没填"，不当成一个值。

    （CLI flag 相反：`--model ""` 是显式给了空值，`pick()` 会照实传递，
    因为命令行上的空串多半是脚本变量没取到，需要立刻炸出来而不是悄悄用默认值。）
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _integer(value: Any) -> int | None:
    return None if value is None else int(value)


def _flag(value: Any) -> bool | None:
    """bool 与"没写"是三件事：`read_only = false` 是显式关掉，不写是没决定。"""
    return None if value is None else bool(value)


def _seconds(value: Any) -> float | None:
    return None if value is None else float(value)


def load_config(path: Path | str | None = None) -> Config:
    """读取配置文件。没有文件就返回全 None 的 Config（所有命令继续用内建默认）。"""
    chosen = config_path(path)
    if chosen is None:
        return Config()
    if not chosen.exists():
        # 显式指的路径不存在要说破：静默当作"没配置"会让人以为参数生效了
        raise ConfigError(f"配置文件不存在: {chosen}")
    try:
        raw = tomllib.loads(chosen.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"配置文件语法错: {chosen}: {exc}") from exc

    unknown: list[str] = []
    problems: list[str] = []

    def read(section: str | None, key: str) -> Any:
        spec = _SCHEMA.get(section, {})
        if key not in spec:
            return None
        table = raw if section is None else raw.get(section, {})
        if not isinstance(table, dict):
            problems.append(f"[{section}] 必须是表（[节] 形式），实际是 {type(table).__name__}")
            return None
        if key not in table:
            return None
        value = table[key]
        problem = _check_type(section, key, value, spec[key])
        if problem:
            problems.append(problem)
            return None
        return value

    def declared() -> None:
        """收集 schema 之外的键与节：写了不生效是最难查的一类失败。"""
        for key in raw:
            if not isinstance(raw[key], dict) and key not in _SCHEMA[None]:
                unknown.append(key)
        for section, table in raw.items():
            if not isinstance(table, dict):
                continue
            spec = _SCHEMA.get(section)
            if spec is None:
                unknown.append(f"[{section}]")
                continue
            for key in table:
                if key not in spec:
                    unknown.append(f"{section}.{key}")

    declared()
    return Config(
        data_dir=_text(read(None, "data_dir")),
        provider=ProviderConfig(
            kind=_text(read("provider", "kind")),
            base_url=_text(read("provider", "base_url")),
            model=_text(read("provider", "model")),
        ),
        gpu=GpuConfig(
            lock_path=_text(read("gpu", "lock_path")),
            stale_after_s=_seconds(read("gpu", "stale_after_s")),
        ),
        retention=RetentionConfig(
            raw_after=_text(read("retention", "raw_after")),
            trace_after=_text(read("retention", "trace_after")),
        ),
        sandbox=SandboxConfig(
            allowed_side_effects=_str_list(read("sandbox", "allowed_side_effects")),
            allowed_impl_prefixes=_str_list(read("sandbox", "allowed_impl_prefixes")),
            default_timeout_ms=_integer(read("sandbox", "default_timeout_ms")),
        ),
        serve=ServeConfig(
            host=_text(read("serve", "host")),
            port=_integer(read("serve", "port")),
            token=_text(read("serve", "token")),
            read_only=_flag(read("serve", "read_only")),
        ),
        alerts=AlertConfig(
            enabled=_flag(read("alerts", "enabled")),
            severities=_str_list(read("alerts", "severities")),
            codes=_str_list(read("alerts", "codes")),
            exclude_codes=_str_list(read("alerts", "exclude_codes")),
            window_s=_integer(read("alerts", "window_s")),
            min_count=_integer(read("alerts", "min_count")),
            cooldown_s=_integer(read("alerts", "cooldown_s")),
            poll_s=_seconds(read("alerts", "poll_s")),
            file=_text(read("alerts", "file")),
        ),
        sinks=_str_list(read("sinks", "events")),
        sources=(chosen,),
        unknown=tuple(unknown),
        problems=tuple(problems),
    )


#: 内建默认。这些值是"部署层"的取舍，所以和配置文件同一处定义——
#: 各条命令的 flag 一律默认 None，由 `pick(flag, cfg.x, 这里的常量)` 决定。
DEFAULT_PROVIDER_KIND = "ollama"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_HOST = "127.0.0.1"
#: 8000 在本机被别的服务占了；这里改成看板真正能起的端口，
#: 而不是让每条命令都带一个 --port 8787 却没人记得。
DEFAULT_PORT = 8787


@dataclass(frozen=True, slots=True)
class Setting:
    """一条"最终会用到的值"以及它是从哪一层来的。"""

    key: str
    value: Any
    source: str  # flag 之外的层：file / env / default


def _layer(file_value: Any, env_value: Any) -> tuple[Any, str]:
    if file_value is not None:
        return file_value, "file"
    if env_value is not None:
        return env_value, "env"
    return None, "default"


def effective(cfg: Config) -> list[Setting]:
    """报告"配置文件 + 环境变量 + 内建默认"三层合出来的生效值。

    这是 `doctor` 那句"写了不生效"的落点：报告里的回落规则必须与各条命令用的
    一模一样，否则它就成了第二个真相。CLI flag 不在这里——它只在被显式给出时赢，
    而 `config show` 无从知道用户接下来会不会带 flag，所以这里报的是"不带 flag 时的值"。

    默认值从各自的家导入而不是抄一份：`tools.sandbox` 在导入时会起线程池，
    所以这两个 import 留在函数里，让 `load_config()` 保持轻量。
    """
    from onyx.eval.gpu_lock import DEFAULT_GPU_STALE_AFTER_S
    from onyx.store.retention import DEFAULT_RAW_AFTER, DEFAULT_TRACE_AFTER
    from onyx.tools.sandbox import DEFAULT_ALLOWED_IMPL_PREFIXES, DEFAULT_TIMEOUT_MS

    rows: list[Setting] = []

    def add(key: str, file_value: Any, env_value: Any, default: Any) -> None:
        value, source = _layer(file_value, env_value)
        rows.append(Setting(key, value if value is not None else default, source))

    add("data_dir", cfg.data_dir, os.environ.get(ENV_DATA_DIR) or None,
        str(project_root() / ".data"))
    add("provider.kind", cfg.provider.kind, None, DEFAULT_PROVIDER_KIND)
    add("provider.base_url", cfg.provider.base_url, None, DEFAULT_BASE_URL)
    add("provider.model", cfg.provider.model, None, None)
    add("gpu.lock_path", cfg.gpu.lock_path, None, None)
    add("gpu.stale_after_s", cfg.gpu.stale_after_s, None, DEFAULT_GPU_STALE_AFTER_S)
    add("retention.raw_after", cfg.retention.raw_after, None, DEFAULT_RAW_AFTER)
    add("retention.trace_after", cfg.retention.trace_after, None, DEFAULT_TRACE_AFTER)
    add("sandbox.allowed_side_effects", cfg.sandbox.allowed_side_effects, None, ("read",))
    add("sandbox.allowed_impl_prefixes", cfg.sandbox.allowed_impl_prefixes, None,
        DEFAULT_ALLOWED_IMPL_PREFIXES)
    add("sandbox.default_timeout_ms", cfg.sandbox.default_timeout_ms, None, DEFAULT_TIMEOUT_MS)
    add("serve.host", cfg.serve.host, None, DEFAULT_HOST)
    add("serve.port", cfg.serve.port, None, DEFAULT_PORT)
    add("serve.read_only", cfg.serve.read_only, None, False)
    # token 只报"有没有设"，值一律不落终端：它会进 shell 历史、CI 日志和截图。
    token_source = next(
        (source for source, value in (
            ("file", cfg.serve.token),
            ("env", os.environ.get(ENV_TOKEN) or None),
        ) if value),
        "default",
    )
    rows.append(Setting(
        "serve.token",
        f"已设置（{len(pick(os.environ.get(ENV_TOKEN) or None, cfg.serve.token) or '')} 字符，值不打印）"
        if token_source != "default" else "未设（仅回环绑定时允许）",
        token_source,
    ))
    add("sinks.events", cfg.sinks, None, ())

    # [alerts] 的默认值来自 rules（它的家），这里只负责报告"哪一层决定了它"
    from onyx.obs.alerts.rules import (
        DEFAULT_ALERT_COOLDOWN_S,
        DEFAULT_ALERT_MIN_COUNT,
        DEFAULT_ALERT_POLL_S,
        DEFAULT_ALERT_SEVERITIES,
        DEFAULT_ALERT_WINDOW_S,
    )

    a = cfg.alerts
    add("alerts.enabled", a.enabled, None, True)
    add("alerts.severities", a.severities, None, DEFAULT_ALERT_SEVERITIES)
    add("alerts.codes", a.codes, None, ())
    add("alerts.exclude_codes", a.exclude_codes, None, ())
    add("alerts.window_s", a.window_s, None, DEFAULT_ALERT_WINDOW_S)
    add("alerts.min_count", a.min_count, None, DEFAULT_ALERT_MIN_COUNT)
    add("alerts.cooldown_s", a.cooldown_s, None, DEFAULT_ALERT_COOLDOWN_S)
    add("alerts.poll_s", a.poll_s, None, DEFAULT_ALERT_POLL_S)
    add("alerts.file", a.file, None, "<数据目录>/alerts/alerts.jsonl")
    return rows


def alert_rule(cfg: Config):
    """把 `[alerts]` 合成**一条**生效规则。

    serve 的后台线程与 `onyx alerts test` 都用它：两处各拼一份参数的话，
    "测试时能发、真出事不发"这种分歧就再也查不动了。
    每个键都按"`None` 才是没给"处理——`min_count = 0`、`severities = []` 是显式意图，
    不能悄悄回落默认值（同 `pick()` 的规矩）。
    """
    from onyx.obs.alerts.rules import (
        DEFAULT_ALERT_COOLDOWN_S,
        DEFAULT_ALERT_MIN_COUNT,
        DEFAULT_ALERT_POLL_S,
        DEFAULT_ALERT_SEVERITIES,
        DEFAULT_ALERT_WINDOW_S,
        AlertRule,
    )

    a = cfg.alerts
    from_file = any(value is not None for value in (
        a.enabled, a.severities, a.codes, a.exclude_codes, a.window_s, a.min_count,
        a.cooldown_s, a.poll_s, a.file,
    ))
    return AlertRule(
        severities=DEFAULT_ALERT_SEVERITIES if a.severities is None else a.severities,
        codes=a.codes or (),
        exclude_codes=a.exclude_codes or (),
        window_s=DEFAULT_ALERT_WINDOW_S if a.window_s is None else a.window_s,
        min_count=DEFAULT_ALERT_MIN_COUNT if a.min_count is None else a.min_count,
        cooldown_s=DEFAULT_ALERT_COOLDOWN_S if a.cooldown_s is None else a.cooldown_s,
        poll_s=DEFAULT_ALERT_POLL_S if a.poll_s is None else a.poll_s,
        enabled=True if a.enabled is None else a.enabled,
        source="file" if from_file else "default",
    )


def alert_file_path(cfg: Config, data_dir: Path):
    """文件出口的落点。没配就用数据目录里那个——本地"一定能落"的那条出口。"""
    from onyx.obs.alerts.channels import default_alert_path

    return Path(cfg.alerts.file) if cfg.alerts.file else default_alert_path(data_dir)


def alert_channels(cfg: Config, data_dir: Path) -> list:
    """装配出口。没配的渠道**不装**，而不是装一个"什么都不发"的渠道：
    后者会让 `alerts ls` 显示"有出口"，而它从来没成功投递过任何东西。
    """
    from onyx.obs.alerts.channels import FileChannel

    channels: list = []
    if alert_rule(cfg).enabled:
        channels.append(FileChannel(alert_file_path(cfg, data_dir)))
    return channels
