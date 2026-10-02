"""内置工具实现。

安全前提：这些函数会被**模型给出的参数**调用，所以每个实现都自己做输入净化，
不依赖上层。`impl_ref` 白名单（`onyx.tools.builtin.`）是第一道闸，
这里是第二道——纵深防御，任何一层单独失效都不至于出事。
"""

from __future__ import annotations

from onyx.tools.builtin.calculator import calculate
from onyx.tools.builtin.echo import boom, echo, slow_echo
from onyx.tools.builtin.fs_read import read_file
from onyx.tools.builtin.time_now import time_now

__all__ = ["boom", "calculate", "echo", "read_file", "slow_echo", "time_now"]
