"""首次启动模型向导（M6）。

打包版双击即用的前提是离线模型已预置。向导在启动时本地扫描模型目录，
缺失时给出放置指引（全程不联网）：

- 显示当前模型根目录与各模型状态；
- 可直接打开模型目录手动拷入；
- 若模型已放在别的位置，可「选择已有模型目录」，路径记录到应用目录下的
  ``models-path.txt``，下次启动自动生效；
- 给出在联网机器上用 ``subextractor-cli fetch-model`` 取模型的命令。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import registry


class ModelWizard(QDialog):
    """模型检测与放置指引对话框（模态）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("模型准备向导")
        self.setMinimumSize(680, 460)
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        intro = QLabel(
            "本工具全程离线运行，需要先把模型放到本机。<br>"
            "模型只需准备一次；准备好后点击「重新检查」。",
            self,
        )
        root.addWidget(intro)

        # 模型状态表（内置的 Silero VAD 不需要用户放置，不列出）
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["模型", "状态", "放置位置 / 说明"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table)

        # 模型根目录
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("模型根目录", self))
        self.root_edit = QLineEdit(self)
        self.root_edit.setReadOnly(True)
        root_row.addWidget(self.root_edit, 1)
        open_btn = QPushButton("打开目录", self)
        open_btn.clicked.connect(self._open_root)
        choose_btn = QPushButton("选择已有模型目录…", self)
        choose_btn.clicked.connect(self._choose_root)
        root_row.addWidget(open_btn)
        root_row.addWidget(choose_btn)
        root.addLayout(root_row)

        guide = QLabel(
            "<b>如何获取模型（在联网的电脑上操作一次）：</b><br>"
            "1. 运行 <code>subextractor-cli fetch-model --size small</code> 获取识别模型；<br>"
            "2. 运行 <code>subextractor-cli fetch-model --kind mt</code> 获取翻译模型；<br>"
            "3. 把生成的整个 <code>models</code> 文件夹复制到本机模型根目录，"
            "或点「选择已有模型目录」直接指向复制来的文件夹。<br>"
            "可选高质量识别档：<code>fetch-model --size large-v3</code>（体积大、建议配 GPU）。",
            self,
        )
        guide.setWordWrap(True)
        root.addWidget(guide)

        self.status_label = QLabel("", self)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        bottom = QHBoxLayout()
        self.recheck_btn = QPushButton("重新检查", self)
        self.recheck_btn.clicked.connect(self.refresh)
        self.close_btn = QPushButton("关闭", self)
        self.close_btn.clicked.connect(self.accept)
        bottom.addWidget(self.recheck_btn)
        bottom.addStretch(1)
        bottom.addWidget(self.close_btn)
        root.addLayout(bottom)

    # ------------------------------------------------------------------ 数据

    def refresh(self) -> None:
        """重新扫描模型目录并刷新表格与状态文案。"""
        root = registry.models_root()
        self.root_edit.setText(str(root))
        self.table.setRowCount(0)

        missing: list[registry.ModelSpec] = []
        for spec in registry.expected_models():
            if spec.kind == "vad":
                continue  # Silero VAD 由 faster-whisper 内置
            available = registry.is_available(spec)
            if not available and spec.required:
                missing.append(spec)
            row = self.table.rowCount()
            self.table.insertRow(row)
            status = "✅ 已就绪" if available else ("❌ 缺失（必需）" if spec.required else "⚪ 未放置（可选）")
            location = f"{root / spec.dir_name}"
            cells = (spec.description.split("：")[0], status, location)
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col < 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter if col == 1 else Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, col, item)

        if missing:
            self.status_label.setText(
                f"<b style='color:#c0392b'>还缺 {len(missing)} 个必需模型</b>，"
                "按上面的指引放置后点「重新检查」。未就绪前仅可浏览界面，无法转换。"
            )
            self.close_btn.setText("稍后再说")
        else:
            self.status_label.setText(
                "<b style='color:#1e8449'>必需模型已全部就绪</b>，可以开始转换了。"
            )
            self.close_btn.setText("开始使用")

    # ------------------------------------------------------------------ 操作

    def _open_root(self) -> None:
        root = registry.models_root()
        try:
            (root / "asr").mkdir(parents=True, exist_ok=True)
            (root / "mt").mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root if root.is_dir() else root.parent)))

    def _choose_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择包含 asr/ 与 mt/ 的模型目录（models）", str(registry.app_dir())
        )
        if not path:
            return
        target = Path(path)
        if not ((target / "asr").is_dir() or (target / "mt").is_dir()):
            self.status_label.setText(
                "<b style='color:#c0392b'>该目录下没有 asr/ 或 mt/ 子目录</b>，"
                "请选择包含模型的 models 文件夹。"
            )
            return
        try:
            registry.save_models_root_override(target)
        except OSError as exc:
            self.status_label.setText(f"<b style='color:#c0392b'>保存失败：</b>{exc}")
            return
        self.refresh()
