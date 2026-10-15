"""命令行入口。

可用命令：
- ``subextractor doctor``        自检环境与离线模型
- ``subextractor fetch-model``   一次性联网把 ASR 模型取到本地（M1）
- ``subextractor run``           视频/音频（文件或目录）→ SRT（M1；目录批处理 M5）
- ``subextractor gui``           启动桌面界面（M4）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core import batch, media
from .core.cache import default_cache_dir
from .core.pipeline import PipelineConfig, run
from .models import registry
from .utils.logging import setup_logging


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"subextractor {__version__}")
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    print(f"subextractor {__version__}")
    print(f"python       : {sys.version.split()[0]}")
    print(f"ffmpeg       : {media.ffmpeg_path() or '未找到（run 必需）'}")
    print(f"模型根目录   : {registry.models_root()}")

    installed = registry.installed_asr_sizes()
    print(f"CUDA 设备    : {registry.cuda_device_count()}")
    print(f"已装 ASR 档位: {', '.join(installed) if installed else '（无）'}")
    try:
        tier = registry.resolve_auto_tier()
        print(f"自动档位     : {tier}｜compute_type={registry.resolve_compute_type()}")
    except registry.MissingModelError:
        print("自动档位     : 无法选择——本地没有 ASR 模型")

    print("模型状态:")
    for spec in registry.expected_models():
        state = "就绪" if registry.is_available(spec) else ("缺失" if spec.required else "可选缺失")
        print(f"  [{state}] {spec.kind:<3} {spec.id:<34} {spec.description}")
    if not installed:
        print("  -> 提示：subextractor fetch-model --size small")
    if not registry.has_content(registry.nllb_model_dir()):
        print("  -> 提示：subextractor fetch-model --kind mt")
    return 0


def _cmd_fetch_model(args: argparse.Namespace) -> int:
    """从 Hugging Face 取模型到本地 models 目录（仅需一次，之后完全离线）。"""
    kind = getattr(args, "kind", "asr")

    if kind == "mt":
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("缺少依赖 huggingface_hub：请先 pip install -e .")
            return 2

        target = registry.nllb_model_dir()
        target.mkdir(parents=True, exist_ok=True)
        print(f"下载 {registry.NLLB_REPO_ID} → {target}")
        snapshot_download(
            repo_id=registry.NLLB_REPO_ID,
            local_dir=str(target),
            allow_patterns=list(registry.NLLB_FILES),
        )
        print("完成：后续 run / GUI 将完全离线使用该目录。")
        return 0

    try:
        from faster_whisper import download_model
    except ImportError:
        print("缺少依赖 faster-whisper：请先 pip install -e .")
        return 2

    target = registry.asr_model_dir(args.size)
    target.mkdir(parents=True, exist_ok=True)
    print(f"下载 faster-whisper-{args.size} → {target}")
    download_model(args.size, output_dir=str(target))
    print("完成：后续 run / GUI 将完全离线使用该目录。")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)

    if not args.input:
        print("缺少 --input 参数")
        return 2

    try:
        files = batch.collect_media_files(Path(args.input))
    except (FileNotFoundError, ValueError) as exc:
        print(f"失败：{exc}")
        return 2

    out_base = Path(args.output_dir) if args.output_dir else None
    cache_dir = None if args.no_resume else default_cache_dir(out_base or files[0].parent)

    def on_progress(stage: str, progress: float) -> None:
        # 尾部补空格：中文进度行变短时清掉上一帧的残留字符
        sys.stdout.write(f"\r{stage} {progress * 100:5.1f}%" + " " * 24)
        sys.stdout.flush()

    if len(files) == 1:  # 单文件：行为与 M1-M4 一致
        config = PipelineConfig(
            input_path=files[0],
            source_lang=args.src,
            target_lang=args.dst,
            asr_model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            translator=args.translator,
            output_dir=out_base,
            cache_dir=cache_dir,
        )
        try:
            result = run(config, on_progress=on_progress)
        except registry.MissingModelError as exc:
            print(f"\n{exc}")
            return 2
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"\n失败：{exc}")
            return 1

        print()
        print(
            f"源语 {result.doc.source_lang}｜档位 {result.asr_model or '-'}"
            f"｜共 {len(result.doc.segments)} 条字幕"
        )
        for path in result.outputs:
            print(f"输出：{path}")
        return 0

    configs = [
        PipelineConfig(
            input_path=file,
            source_lang=args.src,
            target_lang=args.dst,
            asr_model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            translator=args.translator,
            output_dir=out_base,
            cache_dir=cache_dir,
        )
        for file in files
    ]
    total = len(configs)
    resume = "开" if cache_dir else "关"
    print(f"共 {total} 个待处理文件｜并发 {args.jobs}｜断点续跑 {resume}")

    def on_item_progress(index: int, count: int, stage: str, progress: float) -> None:
        if args.jobs <= 1:  # 并发模式下多文件进度会互相覆盖，只报完成行
            on_progress(f"[{index}/{count}] {stage}", progress)

    def on_item_done(item: batch.BatchItemResult, index: int, count: int, _: object) -> None:
        sys.stdout.write("\r" + " " * 60 + "\r")  # 清掉当前文件的进度行
        if item.ok:
            print(f"[{index}/{count}] {item.input_path.name}：完成（{item.cues} 条字幕）")
        else:
            print(f"[{index}/{count}] {item.input_path.name}：失败——{item.error}")

    result = batch.run_batch(
        configs,
        concurrency=args.jobs,
        on_item_progress=on_item_progress,
        on_item_done=on_item_done,
    )

    print()
    if result.cancelled:
        print(f"已取消：完成 {result.total} / {total} 个文件")
    else:
        print(f"批处理结束：成功 {len(result.succeeded)}｜失败 {len(result.failed)}｜共 {total}")

    if result.failed:
        report = (out_base or files[0].parent) / "failed-files.txt"
        lines = [f"{item.input_path}\t{item.error}" for item in result.failed]
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"失败清单：{report}")
        return 1
    return 0


def _cmd_gui(_: argparse.Namespace) -> int:
    try:
        from .ui.app import main as gui_main
    except ImportError:
        print("缺少 GUI 依赖 PySide6：请先 pip install -e .[gui]")
        return 2
    return gui_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subextractor",
        description="离线视频 → 双语外挂字幕（en / fr / ja / ko）",
    )
    parser.add_argument("--version", action="version", version=f"subextractor {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("version", help="打印版本")
    sub.add_parser("doctor", help="自检环境与离线模型")

    p_fetch = sub.add_parser("fetch-model", help="一次性联网获取模型到本地（ASR / MT）")
    p_fetch.add_argument("--kind", default="asr", choices=["asr", "mt"], help="模型类别：asr=Whisper（默认）、mt=NLLB 翻译")
    p_fetch.add_argument("--size", default="small", choices=list(registry.ASR_SIZES), help="ASR 档位（kind=asr 时有效）")

    p_run = sub.add_parser("run", help="视频/音频 → 源语/双语 SRT（输入可为文件或目录）")
    p_run.add_argument("--input", help="输入视频/音频文件，或包含媒体文件的目录（批处理）")
    p_run.add_argument("--src", default="auto", help="源语：auto 或 en / fr / ja / ko / ...")
    p_run.add_argument("--dst", default=None, help="目标语：en / fr / ja / ko（与源语相同则跳过翻译）")
    p_run.add_argument("--translator", default="nllb", choices=["nllb", "argos"], help="翻译引擎（默认 nllb）")
    p_run.add_argument(
        "--model",
        default=registry.AUTO,
        choices=[registry.AUTO, *registry.ASR_SIZES],
        help="ASR 档位；auto = 按算力自动选（默认）",
    )
    p_run.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_run.add_argument(
        "--compute-type",
        default=registry.AUTO,
        help="量化类型；auto = CPU 用 int8、CUDA 用 float16（默认）",
    )
    p_run.add_argument("--output-dir", default=None, help="输出目录，默认与输入同目录")
    p_run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="批处理并发数（默认 1 = 顺序执行并复用模型，多文件时显著更快）",
    )
    p_run.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用 ASR 断点缓存（默认启用：<输出目录>/.subextractor-cache，重跑跳过识别）",
    )
    p_run.add_argument("--verbose", action="store_true", help="输出 DEBUG 日志")

    sub.add_parser("gui", help="启动桌面界面")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "version": _cmd_version,
        "doctor": _cmd_doctor,
        "fetch-model": _cmd_fetch_model,
        "run": _cmd_run,
        "gui": _cmd_gui,
    }
    if args.command is None:
        parser.print_help()
        return 0
    return handlers[args.command](args)
