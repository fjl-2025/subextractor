"""打包版命令行入口（PyInstaller）：doctor / fetch-model / run（批处理）。"""

from __future__ import annotations

import ctypes
import sys


def _enable_utf8_console() -> None:
    """Windows 控制台默认 GBK，切到 UTF-8 代码页，避免中文输出乱码。"""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


if __name__ == "__main__":
    _enable_utf8_console()
    from subextractor.cli import main

    raise SystemExit(main())
