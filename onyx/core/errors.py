"""统一错误族。

规则：**跨进程/跨层边界的失败必须转成这里的类型**，看板才能显示"服务没起"
而不是一段 httpx 堆栈。任何 except 都不许把原始第三方异常直接抛给调用方。
"""

from __future__ import annotations


class OnyxError(Exception):
    """所有 Onyx 错误的基类。"""

    #: 看板展示用的短代码，保持稳定（前端按它分支，不要改字符串）
    code: str = "ONYX_ERROR"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict = detail or {}


# ── Provider / 网络 ────────────────────────────────────────────────
class ProviderError(OnyxError):
    code = "PROVIDER_ERROR"


class ProviderUnreachable(ProviderError):
    code = "PROVIDER_UNREACHABLE"

    def __init__(self, message: str, *, base_url: str = "", elapsed_ms: float | None = None, **kw) -> None:
        super().__init__(message, detail={"base_url": base_url, "elapsed_ms": elapsed_ms, **kw})
        self.base_url = base_url


class ProviderRejected(ProviderError):
    """引擎返回了错误体（如 model not found、context length exceeded）。"""

    code = "PROVIDER_REJECTED"

    def __init__(self, message: str, *, status: int | None = None, body: str = "", **kw) -> None:
        super().__init__(message, detail={"status": status, "body": body[:4000], **kw})


class RequestTimeout(ProviderError):
    code = "REQUEST_TIMEOUT"


class CapabilityMissing(OnyxError):
    """请求的能力该 provider 不支持（如 Ollama 的 tool_choice / n>1）。"""

    code = "CAPABILITY_MISSING"


# ── 数据契约 ───────────────────────────────────────────────────────
class SchemaInvalid(OnyxError):
    code = "SCHEMA_INVALID"


class UnknownEventType(SchemaInvalid):
    code = "UNKNOWN_EVENT_TYPE"


# ── 存储 ───────────────────────────────────────────────────────────
class StoreError(OnyxError):
    code = "STORE_ERROR"


class MigrationError(StoreError):
    code = "MIGRATION_ERROR"


class BlobNotFound(StoreError):
    code = "BLOB_NOT_FOUND"


class InvalidBlobRef(StoreError):
    code = "INVALID_BLOB_REF"


# ── 工具 ───────────────────────────────────────────────────────────
class ToolError(OnyxError):
    code = "TOOL_ERROR"


class ToolUnknown(ToolError):
    code = "TOOL_UNKNOWN"


class ToolArgError(ToolError):
    """参数非法。契约测试的关键断言：这类错误不许变成进程崩溃或 200。"""

    code = "TOOL_ARG_ERROR"


class ToolTimeout(ToolError):
    code = "TOOL_TIMEOUT"


class ToolSandboxDenied(ToolError):
    code = "TOOL_SANDBOX_DENIED"


class ToolRuntime(ToolError):
    code = "TOOL_RUNTIME"


# ── 评测 ───────────────────────────────────────────────────────────
class EvalError(OnyxError):
    code = "EVAL_ERROR"


class DatasetNotFound(EvalError):
    code = "DATASET_NOT_FOUND"


class TaskSkipped(EvalError):
    """能力不匹配导致的跳过。必须带原因，禁止静默降级（DESIGN §9.1）。"""

    code = "TASK_SKIPPED"

    def __init__(self, message: str, *, reason: str = "", missing: tuple[str, ...] = ()) -> None:
        super().__init__(message, detail={"reason": reason, "missing": list(missing)})
        self.reason = reason


class BudgetExceeded(EvalError):
    code = "BUDGET_EXCEEDED"


class GpuLockBusy(OnyxError):
    """本地 GPU 是独占资源：eval / bench / playground 互斥（DESIGN §8.5）。"""

    code = "GPU_LOCK_BUSY"
