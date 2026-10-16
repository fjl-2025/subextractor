"""桌面界面（M4；M5 起支持目录批处理，PySide6）。

布局（自上而下）：
1. 输入/输出：文件或目录（拖拽或浏览，目录 = 批处理）、输出目录（可选，默认与输入同目录）
2. 选项：源语、目标语、ASR 档位、翻译引擎、是否输出 ASS
3. 操作：开始 / 取消 / 打开输出目录，进度条 + 分阶段状态
4. 结果：postcheck 提示 + 字幕表格预览

职责边界：GUI 只做交互与展示，流程逻辑全部在 core.pipeline；
管线在 QThread 中运行，进度/结果/错误经信号回传主线程，界面不冻结。
取消通过 ``should_cancel`` 注入管线，取消时临时音轨随临时目录自动清理。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..config import SUPPORTED_TARGET_LANGS
from ..core import batch, ocr, pipeline
from ..core.cache import default_cache_dir
from ..core.errors import PipelineCancelled
from ..core.pipeline import PipelineConfig, PipelineResult
from ..core.subtitle import SubtitleDoc
from ..models import registry
from ..utils.logging import setup_logging
from .wizard import ModelWizard

MEDIA_FILTER = "视频/音频 (*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.m4a *.wav *.flac *.aac *.ogg);;所有文件 (*.*)"

SOURCE_LANGS: tuple[str, ...] = ("auto", *SUPPORTED_TARGET_LANGS)
LANG_LABELS = {
    "auto": "自动检测",
    "zh": "中文（简体）",
    "en": "英语 English",
    "fr": "法语 Français",
    "ja": "日语 日本語",
    "ko": "韩语 한국어",
}


def _readable_error(exc: Exception) -> str:
    """界面可读文案；core 层抛出的异常本身已是中文。"""
    return str(exc).strip() or f"处理失败：{exc.__class__.__name__}"


def _fmt_clock(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds >= 3600:
        hours, rem = divmod(int(seconds), 3600)
        return f"{hours}:{rem // 60:02d}:{rem % 60:02d}"
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


class PipelineWorker(QThread):
    """在独立线程执行管线；进度/结果/错误通过信号回传主线程。"""

    stage_changed = Signal(str, float)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, config: PipelineConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """请求取消：管线在最近的检查点抛 PipelineCancelled。"""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = pipeline.run(
                self._config,
                on_progress=self.stage_changed.emit,
                should_cancel=self._cancel_event.is_set,
            )
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 —— 一切异常都以可读文案回到界面
            self.failed.emit(_readable_error(exc))
        else:
            self.succeeded.emit(result)


class BatchWorker(QThread):
    """目录批处理：独立线程逐文件跑管线，逐项回报状态（M5）。"""

    item_started = Signal(int, int, str)  # (序号 1 起, 总数, 文件名)
    item_done = Signal(object, int, int, object)  # (BatchItemResult, 序号, 总数, SubtitleDoc|None)
    stage_changed = Signal(str, float)
    succeeded = Signal(object)  # BatchResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, configs: list[PipelineConfig], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configs = configs
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """请求取消：当前文件跑完或中止后，不再开始下一个。"""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = batch.run_batch(
                self._configs,
                should_cancel=self._cancel_event.is_set,
                on_item_started=lambda index, total, path: self.item_started.emit(
                    index, total, path.name
                ),
                on_item_progress=lambda index, total, stage, progress: self.stage_changed.emit(
                    f"[{index}/{total}] {stage}", (index - 1 + progress) / total
                ),
                on_item_done=lambda item, index, total, doc: self.item_done.emit(
                    item, index, total, doc
                ),
            )
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(_readable_error(exc))
        else:
            if result.cancelled:
                self.cancelled.emit()
            else:
                self.succeeded.emit(result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"SubExtractor {__version__} — 离线双语字幕")
        self.setAcceptDrops(True)
        self.resize(900, 660)

        self._settings = QSettings("subextractor", "subextractor")
        self._worker: PipelineWorker | BatchWorker | None = None
        self._running = False
        self._result: PipelineResult | None = None
        self._open_dir: Path | None = None
        self._preview_doc: SubtitleDoc | None = None

        self._build_ui()
        self._build_menu()
        self._refresh_model_status()  # 先填充档位下拉，_load_settings 才能选中上次档位
        self._load_settings()
        self._update_output_hint()
        QTimer.singleShot(150, self._maybe_show_wizard)  # 界面显示后再做首启模型检查

    def _update_output_hint(self) -> None:
        """目标语下拉下方的产物说明：选目标语即自动出双语，无需额外开关。"""
        if self.dst_combo.currentData():
            self.output_hint.setText(
                "选定目标语后自动产出三个文件：源语 .orig.srt、纯译文 .<语言>.srt、"
                "双语对照（与视频同名 .srt，播放器打开视频即自动外挂）。"
            )
        else:
            self.output_hint.setText("仅输出源语字幕，不翻译（只生成 .orig.srt）。")

    # ------------------------------------------------------------------ 菜单

    def _build_menu(self) -> None:
        help_menu = self.menuBar().addMenu("帮助")
        wizard_action = help_menu.addAction("模型准备向导…")
        wizard_action.triggered.connect(self._open_wizard)

    # ------------------------------------------------------------------ UI 构建

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)

        # 输入 / 输出
        io_box = QGroupBox("输入 / 输出", central)
        io = QGridLayout(io_box)
        self.input_edit = QLineEdit(io_box)
        self.input_edit.setPlaceholderText("把视频/音频文件或整个文件夹拖到这里，或点击右侧浏览")
        input_btn = QPushButton("浏览…", io_box)
        input_btn.clicked.connect(self._choose_input)
        dir_btn = QPushButton("目录…", io_box)
        dir_btn.clicked.connect(self._choose_input_dir)
        io.addWidget(QLabel("输入", io_box), 0, 0)
        io.addWidget(self.input_edit, 0, 1)
        io.addWidget(input_btn, 0, 2)
        io.addWidget(dir_btn, 0, 3)

        self.outdir_edit = QLineEdit(io_box)
        self.outdir_edit.setPlaceholderText("留空 = 输出到视频旁边（推荐：与视频同名可自动外挂）")
        outdir_btn = QPushButton("浏览…", io_box)
        outdir_btn.clicked.connect(self._choose_outdir)
        io.addWidget(QLabel("输出目录", io_box), 1, 0)
        io.addWidget(self.outdir_edit, 1, 1)
        io.addWidget(outdir_btn, 1, 2)
        root.addWidget(io_box)

        # 选项
        opt_box = QGroupBox("选项", central)
        opts = QFormLayout(opt_box)

        self.src_combo = QComboBox(opt_box)
        for lang in SOURCE_LANGS:
            self.src_combo.addItem(LANG_LABELS.get(lang, lang), lang)

        self.dst_combo = QComboBox(opt_box)
        self.dst_combo.addItem("（仅源语字幕）", "")
        for lang in SUPPORTED_TARGET_LANGS:
            self.dst_combo.addItem(LANG_LABELS.get(lang, lang), lang)

        self.model_combo = QComboBox(opt_box)

        self.engine_combo = QComboBox(opt_box)
        self.engine_combo.addItem("NLLB-200（主力）", "nllb")
        self.engine_combo.setItemData(
            0, "Argos 轻量备选引擎计划后续接入", Qt.ItemDataRole.ToolTipRole
        )

        self.ass_check = QCheckBox("同时输出 ASS（源语/译文分样式）", opt_box)
        self.ocr_check = QCheckBox("画面硬字幕（OCR 提取，跳过语音识别）", opt_box)
        self.ocr_check.setToolTip("用于无音轨/画面内嵌字幕的视频；需安装 OCR 扩展并预置模型")

        glossary_row = QHBoxLayout()
        self.glossary_edit = QLineEdit(opt_box)
        self.glossary_edit.setPlaceholderText(
            "术语表（可选，UTF-8 TSV：源词<TAB>目标词），翻译后对译文替换"
        )
        glossary_btn = QPushButton("浏览…", opt_box)
        glossary_btn.clicked.connect(self._choose_glossary)
        glossary_row.addWidget(self.glossary_edit, 1)
        glossary_row.addWidget(glossary_btn)
        opts.addRow("源语", self.src_combo)
        opts.addRow("目标语", self.dst_combo)
        self.output_hint = QLabel("", opt_box)
        self.output_hint.setWordWrap(True)
        self.output_hint.setStyleSheet("color: #666; font-size: 12px;")
        self.dst_combo.currentIndexChanged.connect(self._update_output_hint)
        opts.addRow("", self.output_hint)
        opts.addRow("ASR 档位", self.model_combo)
        opts.addRow("翻译引擎", self.engine_combo)
        opts.addRow("", self.ass_check)
        opts.addRow("", self.ocr_check)
        opts.addRow("术语表", glossary_row)
        root.addWidget(opt_box)

        # 操作 + 进度
        btns = QHBoxLayout()
        self.start_btn = QPushButton("开始转换", central)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("取消", central)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.open_btn = QPushButton("打开输出目录", central)
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_output_dir)
        self.preview_btn = QPushButton("预览对照字幕", central)
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._preview_subtitles)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch(1)
        btns.addWidget(self.preview_btn)
        btns.addWidget(self.open_btn)
        root.addLayout(btns)

        self.stage_label = QLabel("就绪", central)
        self.progress = QProgressBar(central)
        self.progress.setRange(0, 100)
        root.addWidget(self.stage_label)
        root.addWidget(self.progress)

        # 校验提示 + 结果表格
        self.notes = QPlainTextEdit(central)
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(110)
        self.notes.hide()
        root.addWidget(self.notes)

        self.table = QTableWidget(0, 5, central)
        self.table.setHorizontalHeaderLabels(["#", "开始", "结束", "源语", "译文"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        root.addWidget(self.table, 1)

        self.setCentralWidget(central)
        self.statusBar().addPermanentWidget(QLabel("", self))

    def _refresh_model_status(self) -> None:
        """档位下拉只列本地已装项；状态栏常驻模型提示。"""
        installed = registry.installed_asr_sizes()
        current = self.model_combo.currentData() if self.model_combo.currentText() else None
        self.model_combo.clear()
        try:
            tier = registry.resolve_auto_tier()
            auto_label = f"自动（当前：{tier}）"
        except registry.MissingModelError:
            auto_label = "自动（本地无 ASR 模型）"
        self.model_combo.addItem(auto_label, registry.AUTO)
        for size in registry.ASR_SIZES:
            if size in installed:
                self.model_combo.addItem(size, size)
        if current is not None:
            idx = self.model_combo.findData(current)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

        hint = self.statusBar().findChild(QLabel)
        if hint is not None:
            if not installed:
                hint.setText("缺少 ASR 模型：subextractor fetch-model --size small")
            elif not registry.has_content(registry.nllb_model_dir()):
                hint.setText("翻译模型缺失，仅可输出源语字幕：subextractor fetch-model --kind mt")
            else:
                hint.setText("")

    # ------------------------------------------------------------------ 拖拽与选择

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._running:
            return
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            self.input_edit.setText(urls[0].toLocalFile())

    def _choose_input(self) -> None:
        if self._running:
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择视频/音频", "", MEDIA_FILTER)
        if path:
            self.input_edit.setText(path)

    def _choose_input_dir(self) -> None:
        if self._running:
            return
        path = QFileDialog.getExistingDirectory(self, "选择包含视频/音频的目录（批处理）")
        if path:
            self.input_edit.setText(path)

    def _choose_outdir(self) -> None:
        if self._running:
            return
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.outdir_edit.setText(path)

    def _choose_glossary(self) -> None:
        if self._running:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择术语表（TSV：源词<TAB>目标词）", "", "术语表 (*.tsv *.txt);;所有文件 (*)"
        )
        if path:
            self.glossary_edit.setText(path)

    # ------------------------------------------------------------------ 运行控制

    def _start(self) -> None:
        if self._running:
            return
        input_text = self.input_edit.text().strip()
        if not input_text:
            QMessageBox.warning(self, "缺少输入", "请先选择要处理的视频/音频文件或目录。")
            return

        try:
            files = batch.collect_media_files(Path(input_text))
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.warning(self, "输入无效", str(exc))
            return

        outdir_text = self.outdir_edit.text().strip()
        outdir = Path(outdir_text) if outdir_text else None
        glossary_text = self.glossary_edit.text().strip()
        if glossary_text and not Path(glossary_text).is_file():
            QMessageBox.warning(self, "术语表无效", f"文件不存在：{glossary_text}")
            return
        ocr_config = None
        if self.ocr_check.isChecked():
            if not ocr.ocr_available():
                QMessageBox.warning(
                    self,
                    "OCR 扩展未安装",
                    "画面硬字幕需要 OCR 扩展：pip install -e .[ocr]\n"
                    "模型预置：subextractor fetch-model --kind ocr --lang ja",
                )
                return
            ocr_config = ocr.OcrConfig(lang=self.src_combo.currentData() or "auto")
        common = {
            "source_lang": self.src_combo.currentData() or "auto",
            "target_lang": self.dst_combo.currentData() or None,
            "asr_model": self.model_combo.currentData() or registry.AUTO,
            "translator": self.engine_combo.currentData() or "nllb",
            "output_dir": outdir,
            "emit_ass": self.ass_check.isChecked(),
            "glossary_path": Path(glossary_text) if glossary_text else None,
            "cache_dir": default_cache_dir(outdir or files[0].parent),
            "ocr_config": ocr_config,
        }
        configs = [PipelineConfig(input_path=file, **common) for file in files]
        self._open_dir = outdir or files[0].parent
        self._save_settings()

        self._running = True
        self._set_inputs_enabled(False)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.progress.setValue(0)
        self.stage_label.setText("准备中…")
        self._result = None
        self._preview_doc = None
        self.table.setRowCount(0)
        if len(configs) > 1:
            self.notes.setPlainText("")
            self.notes.show()
        else:
            self.notes.hide()

        if len(configs) > 1:
            self._worker = BatchWorker(configs, parent=self)
            self._worker.item_started.connect(self._on_item_started)
            self._worker.item_done.connect(self._on_item_done)
            self._worker.succeeded.connect(self._on_batch_success)
        else:
            self._worker = PipelineWorker(configs[0], parent=self)
            self._worker.succeeded.connect(self._on_success)
        self._worker.stage_changed.connect(self._on_stage)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self.stage_label.setText("正在取消…")
            self.cancel_btn.setEnabled(False)
            self._worker.cancel()

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for widget in (
            self.input_edit,
            self.outdir_edit,
            self.src_combo,
            self.dst_combo,
            self.model_combo,
            self.engine_combo,
            self.ass_check,
        ):
            widget.setEnabled(enabled)

    # ------------------------------------------------------------------ 结果槽

    def _on_stage(self, stage: str, progress: float) -> None:
        self.stage_label.setText(stage)
        self.progress.setValue(int(progress * 100))

    def _on_success(self, result: object) -> None:
        assert isinstance(result, PipelineResult)
        self._result = result
        self.stage_label.setText("完成")
        self.progress.setValue(100)
        self._show_issues(result.issues)
        self._fill_table(result)
        self.open_btn.setEnabled(True)
        summary = (
            f"源语 {result.doc.source_lang}｜档位 {result.asr_model or '-'}"
            f"｜共 {len(result.doc.segments)} 条字幕"
        )
        if result.doc.target_lang:
            summary += f"｜目标语 {result.doc.target_lang}"
        if result.main_output is not None:
            summary += f"｜双语成品 {result.main_output.name}（与视频同名，直接外挂）"
        self.statusBar().showMessage(summary)

    def _on_failed(self, message: str) -> None:
        self.stage_label.setText("失败")
        QMessageBox.critical(self, "转换失败", message)

    def _on_cancelled(self) -> None:
        self.stage_label.setText("已取消（临时文件已清理）")
        self.statusBar().showMessage("任务已取消", 3000)

    # ------------------------------------------------------------------ 批处理槽

    def _on_item_started(self, index: int, total: int, name: str) -> None:
        self.stage_label.setText(f"[{index}/{total}] {name}")

    def _on_item_done(self, item: object, index: int, total: int, doc: object) -> None:
        assert isinstance(item, batch.BatchItemResult)
        name = item.input_path.name
        if item.ok:
            self.notes.appendPlainText(f"[{index}/{total}] 完成：{name}（{item.cues} 条字幕）")
            if isinstance(doc, SubtitleDoc):
                self._fill_table_doc(doc)
        else:
            self.notes.appendPlainText(f"[{index}/{total}] 失败：{name}——{item.error}")
        self.progress.setValue(int(index / total * 100))

    def _on_batch_success(self, result: object) -> None:
        assert isinstance(result, batch.BatchResult)
        self.stage_label.setText("完成")
        self.progress.setValue(100)
        self.open_btn.setEnabled(True)
        summary = f"批处理完成：成功 {len(result.succeeded)}｜失败 {len(result.failed)}"
        self.notes.appendPlainText(summary)
        self.statusBar().showMessage(summary)

    def _on_finished(self) -> None:
        self._running = False
        self._set_inputs_enabled(True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _show_issues(self, issues: list) -> None:
        if not issues:
            self.notes.hide()
            return
        errors = sum(1 for issue in issues if issue.level == "error")
        lines = [f"输出校验：{len(issues)} 项提示（含 {errors} 项错误）"]
        lines += [
            f"[{'错误' if issue.level == 'error' else '警告'}] #{issue.index} {issue.message}"
            for issue in issues[:20]
        ]
        if len(issues) > 20:
            lines.append(f"… 其余 {len(issues) - 20} 项见日志")
        self.notes.setPlainText("\n".join(lines))
        self.notes.show()

    def _fill_table(self, result: PipelineResult) -> None:
        self._fill_table_doc(result.doc)

    def _fill_table_doc(self, doc: SubtitleDoc) -> None:
        segments = doc.segments
        self._preview_doc = doc
        self.preview_btn.setEnabled(bool(segments))
        self.table.setRowCount(len(segments))
        for row, seg in enumerate(segments):
            values = (
                str(seg.index),
                _fmt_clock(seg.start),
                _fmt_clock(seg.end),
                seg.text,
                seg.translation or "",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col < 3:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)

    def _open_output_dir(self) -> None:
        if self._open_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._open_dir)))
        elif self._result and self._result.outputs:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._result.outputs[0].parent)))

    def _preview_subtitles(self) -> None:
        """转换完成后现场对照：弹窗按最终排版展示源语/译文全文（不截断）。"""
        doc = self._preview_doc
        if doc is None or not doc.segments:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("字幕对照预览（与写出文件内容一致）")
        dlg.resize(860, 640)
        layout = QVBoxLayout(dlg)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("显示：", dlg))
        mode = QComboBox(dlg)
        mode.addItem("双语对照", "bilingual")
        mode.addItem("仅源语", "src")
        if doc.target_lang:
            mode.addItem("仅译文", "dst")
        bar.addWidget(mode)
        bar.addStretch(1)
        count_lbl = QLabel(f"{len(doc.segments)} 条", dlg)
        bar.addWidget(count_lbl)
        layout.addLayout(bar)

        view = QPlainTextEdit(dlg)
        view.setReadOnly(True)
        mono = view.font()
        mono.setFamily("Consolas")
        view.setFont(mono)
        layout.addWidget(view, 1)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        close_btn = QPushButton("关闭", dlg)
        close_btn.clicked.connect(dlg.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

        def render() -> None:
            m = mode.currentData()
            blocks: list[str] = []
            for seg in doc.segments:
                header = f"#{seg.index}  {_fmt_clock(seg.start)} → {_fmt_clock(seg.end)}"
                if m == "src":
                    body = seg.text
                elif m == "dst":
                    body = seg.translation or ""
                else:  # bilingual：与双语 SRT 同样的上下排列
                    body = seg.text + (f"\n{seg.translation}" if seg.translation else "")
                blocks.append(f"{header}\n{body}")
            view.setPlainText("\n\n".join(blocks))

        mode.currentIndexChanged.connect(render)
        render()
        dlg.exec()

    # ------------------------------------------------------------------ 配置持久化

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("input", self.input_edit.text().strip())
        s.setValue("output_dir", self.outdir_edit.text().strip())
        s.setValue("source_lang", self.src_combo.currentData())
        s.setValue("target_lang", self.dst_combo.currentData())
        s.setValue("asr_model", self.model_combo.currentData())
        s.setValue("engine", self.engine_combo.currentData())
        s.setValue("emit_ass", self.ass_check.isChecked())
        s.setValue("use_ocr", self.ocr_check.isChecked())
        s.setValue("glossary", self.glossary_edit.text().strip())

    def _load_settings(self) -> None:
        s = self._settings
        self.input_edit.setText(str(s.value("input") or ""))
        self.outdir_edit.setText(str(s.value("output_dir") or ""))
        for combo, key in (
            (self.src_combo, "source_lang"),
            (self.dst_combo, "target_lang"),
            (self.model_combo, "asr_model"),
            (self.engine_combo, "engine"),
        ):
            value = s.value(key)
            if value is not None:
                idx = combo.findData(value)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        self.ass_check.setChecked(s.value("emit_ass", False, type=bool))
        self.ocr_check.setChecked(s.value("use_ocr", False, type=bool))
        self.glossary_edit.setText(str(s.value("glossary") or ""))

    # ------------------------------------------------------------------ 首启向导

    def _maybe_show_wizard(self) -> None:
        """必需模型缺失时（典型为干净机器首次启动）自动弹出向导。"""
        if registry.missing_models():
            self._open_wizard()

    def _open_wizard(self) -> None:
        dialog = ModelWizard(self)
        dialog.exec()
        self._refresh_model_status()  # 用户可能在向导里换了模型目录，刷新档位与状态栏

    # ------------------------------------------------------------------ 退出

    def closeEvent(self, event: QCloseEvent) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            answer = QMessageBox.question(
                self,
                "任务进行中",
                "转换尚未完成，退出将中断任务并清理临时文件。确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            worker.cancel()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            worker.wait()  # 模型加载中无法秒断，等到检查点触发取消
            QApplication.restoreOverrideCursor()
        event.accept()


def main() -> int:
    setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("SubExtractor")
    window = MainWindow()
    window.show()
    return app.exec()
