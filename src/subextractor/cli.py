"""命令行入口。

M0 阶段可用命令：
- ``subextractor --version``  打印版本
- ``subextractor doctor``     自检环境与离线模型
其余子命令为占位，按迭代计划逐步填充。
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .core import media
from .models import registry


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"subextractor {__version__}")
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    print(f"subextractor {__version__}")
    print(f"python       : {sys.version.split()[0]}")
    print(f"ffmpeg       : {media.ffmpeg_path() or '未找到（M1 前不影响）'}")
    print(f"模型根目录   : {registry.models_root()}")
    print("模型状态:")
    for spec in registry.expected_models():
        state = "就绪" if registry.is_available(spec) else ("缺失" if spec.required else "可选缺失")
        print(f"  [{state}] {spec.kind:<3} {spec.id:<34} {spec.description}")
    if not registry.models_root().exists():
        print("  -> 提示：请按 docs/design-outline.md §10 放置离线模型")
    return 0


def _cmd_run(_: argparse.Namespace) -> int:
    print("run 尚未实现（计划 M1）：音轨抽取 → 源语识别 → 翻译 → 双语字幕")
    return 1


def _cmd_gui(_: argparse.Namespace) -> int:
    print("gui 尚未实现（计划 M4）")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subextractor",
        description="离线视频 → 双语外挂字幕（en / fr / ja / ko）",
    )
    parser.add_argument("--version", action="version", version=f"subextractor {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="打印版本").set_defaults()
    sub.add_parser("doctor", help="自检环境与离线模型")

    p_run = sub.add_parser("run", help="执行转换（M1 起可用）")
    p_run.add_argument("--input", help="输入视频/音频文件")
    p_run.add_argument("--src", default="auto", help="源语，auto 为自动检测")
    p_run.add_argument("--dst", default=None, help="目标语：en / fr / ja / ko；留空只出源语字幕")

    sub.add_parser("gui", help="启动桌面界面（M4 起可用）")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "version": _cmd_version,
        "doctor": _cmd_doctor,
        "run": _cmd_run,
        "gui": _cmd_gui,
    }
    if args.command is None:
        parser.print_help()
        return 0
    return handlers[args.command](args)
