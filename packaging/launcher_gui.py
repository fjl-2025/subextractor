"""打包版 GUI 入口（PyInstaller）：双击直接进入桌面界面，无控制台窗口。"""

from __future__ import annotations

from subextractor.ui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
