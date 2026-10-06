""""这个数是哪个版本跑出来的"——运行自身的出处信息。

和 `clock.py` / `ids.py` 同一类：它们不参与业务，但没有它们，落库的每个数字都会缺一句
"我来自哪里"。`eval_run.app_version/git_rev` 与 `perf_run` 的指纹都靠这个锚点回答
"这次和上次是不是同一份代码"，而配对回归的前提就是这句话有答案。
"""

from __future__ import annotations


def git_rev(timeout_s: float = 3.0) -> str:
    """当前 commit 的短 hash。拿不到就返回空串——**空串是"不知道"，不是"没有版本"**。

    渲染成「—」而不是 0 或 `unknown` 之外的猜测值。非 git 环境（装成 wheel 后跑在生产机上）
    本来就没有 commit，这时版本靠 `app_version` 兜底。
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=timeout_s, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        # OSError 覆盖"找不到 git"；SubprocessError 覆盖超时。两种都只是"不知道版本"，
        # 不该让一条测量命令因为版本问不出来就失败
        return ""
