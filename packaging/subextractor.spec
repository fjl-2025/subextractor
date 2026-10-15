# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（M6，Windows onedir）。

产物：dist/SubExtractor/
    SubExtractor.exe       双击启动的 GUI（无控制台）
    subextractor-cli.exe   命令行（doctor / fetch-model / run 批处理）
    _internal/             运行时与依赖（build 脚本另建同级 models/ 放模型）

模型权重不进包（ASR 数 GB、NLLB 约 600MB），由用户预置到 exe 同级 models/，
或首启向导指向已复制好的模型目录。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

HERE = Path(SPECPATH).resolve()
ROOT = HERE.parent
SRC = ROOT / "src"

datas = [
    (str(SRC / "subextractor" / "config" / "languages.yaml"), "subextractor/config"),
]
binaries = []
hiddenimports: list[str] = []

# 带数据文件/原生库、静态分析容易漏的包：整体收集
for package in (
    "janome",          # 日文分词词典（sysdic 数据）
    "faster_whisper",  # 内置 Silero VAD（assets/silero_vad_v6.onnx）
    "imageio_ffmpeg",  # 内置 ffmpeg 可执行文件
    "onnxruntime",     # Silero VAD 推理原生库
    "ctranslate2",     # ASR / NLLB 推理引擎原生库
    "tokenizers",      # NLLB tokenizer 原生库
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# transformers 大量类经动态注册加载（AutoTokenizer），静态分析会漏
hiddenimports += collect_submodules("transformers")
datas += collect_data_files("transformers")

# 确认不会用到的大件，减小体积
excludes = [
    "tkinter",
    "matplotlib",
    "pandas",
    "pytest",
    "IPython",
    "notebook",
    "jupyter",
]

common = dict(
    pathex=[str(SRC), str(HERE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

a_gui = Analysis(["launcher_gui.py"], name="SubExtractor", **common)
a_cli = Analysis(["launcher_cli.py"], name="subextractor-cli", **common)

pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)

exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="SubExtractor",
    console=False,
    disable_windowed_traceback=False,
)
exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="subextractor-cli",
    console=True,
)

coll = COLLECT(
    exe_gui,
    exe_cli,
    a_gui.binaries,
    a_gui.zipfiles,
    a_gui.datas,
    a_cli.binaries,
    a_cli.zipfiles,
    a_cli.datas,
    name="SubExtractor",
)
