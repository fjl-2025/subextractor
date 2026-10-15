"""核心层共享异常。

独立成模块的原因：asr / translate / pipeline 相互依赖，
取消异常放在任一使用方里都会造成循环导入。
"""

from __future__ import annotations


class PipelineCancelled(RuntimeError):
    """任务被用户取消（GUI 取消按钮触发）。"""
