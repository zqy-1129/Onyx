"""扩展点契约测试的公共装置。

`plugins_example/` 的两个样板包由根 conftest 挂进 `sys.path`（它们是外部插件，
不是 onyx 的依赖），所以这里可以直接用伪造的 entry point 指向真实源码。

真正的"装上插件再跑"用
`uv run --with-editable plugins_example/example_task onyx eval tasks` 手工验证
（见 docs/IMPLEMENTATION.md S16 自测）——CI 不能依赖安装步骤，
否则扩展点测试会退化成一个"看运气"的检查。
"""

from __future__ import annotations
