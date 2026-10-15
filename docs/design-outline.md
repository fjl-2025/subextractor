# SubExtractor 设计大纲

> 离线桌面工具：把既有视频/音频转成带时间轴的字幕，并生成目标语翻译，输出可外挂播放的双语字幕文件。
> 全程本地运行，不联网（模型需事先预置到本地）。

输入视频 → 输出 `name.bilingual.<lang>.srt` / `.ass`

---

## 1. 目标与范围

### 1.1 范围内（In scope）
- 输入：本地视频/音频文件（mp4 / mkv / mov / mp3 / m4a / wav 等）
- 识别：源语自动检测，或手动指定源语
- 翻译：**英语（en） / 法语（fr） / 日语（ja） / 韩语（ko）** 互译（源语 ≠ 目标语时执行）
- 输出：单语 SRT、双语 SRT、ASS、VTT
- 桌面 GUI：拖拽、批量、进度、预览、导出
- 全流程离线

### 1.2 非目标（Out of scope，首版不做）
- 实时 / 直播字幕
- 配音（TTS）与视频重新合成
- 云端 API、任何联网依赖
- 移动端
- 画面硬字幕 OCR（列为 M6 可选扩展）

---

## 2. 使用场景
| 场景 | 描述 | 典型操作 |
|---|---|---|
| A 单个视频 | 一个视频 → 双语 SRT → 播放器外挂 | 拖入文件，选目标语，导出 |
| B 批量剧集 | 一个目录多文件 → 批量输出 | 选目录，队列跑，结果落盘 |
| C 只要源语 | 仅转写不翻译 | 目标语留空 |

---

## 3. 多语种策略（en / fr / ja / ko）

源语由 Whisper 覆盖（可自动检测或强制指定）；目标语限定为上述四语。
**断句与排版参数按语言区分**，这是质量的关键差异点：

| 语言 | 分词方式 | 建议每行字符数 | 每屏行数 | 备注 |
|---|---|---|---|---|
| en | 空格分词 | ≤ 42 字符 | 2 | 常规拉丁断行 |
| fr | 空格分词 | ≤ 42 字符 | 2 | 注意连字符、省音（l'/d'）不孤立 |
| ja | 无空格（按字符/文节） | ≤ 13 全角字符 | 2 | 避免助词（は/が/を…）开头断行 |
| ko | 空格分词（어절） | ≤ 16 字符 | 2 | 助词不与词干拆开 |

> 上述字符数为业界惯例（Netflix 风格）参考值，M3 阶段用真实样本实测校准后写入语言参数表。

---

## 4. 技术选型

| 层 | 选型 | 备选 | 理由 |
|---|---|---|---|
| 语言/运行时 | Python 3.10+ | — | 生态最全，四个语言模型都有现成绑定 |
| GUI | PySide6 (Qt6) | Tauri+Rust / customtkinter | 拖拽、进度、跨平台、打包成熟 |
| 音轨抽取 | FFmpeg（imageio-ffmpeg 内置二进制） | 系统 FFmpeg | 免用户安装，统一行为 |
| ASR | faster-whisper (CTranslate2) | whisper.cpp / WhisperX | int8 量化、CPU/GPU 均可、多语种 |
| 时间轴增强 | WhisperX（词级对齐+VAD） | stable-ts / whisper-timestamped | 词级时间戳决定断句质量 |
| 翻译 | NLLB-200-distilled-600M（CTranslate2 int8） | Argos Translate | NLLB 覆盖 200 语含四语，质量统一；Argos 轻量但对 ja/ko 弱、多经英语中转 |
| 字幕 IO | pysrt / srt / ass | — | 读写 SRT/VTT/ASS |
| 配置 | PyYAML | JSON | 语言参数表 + 用户配置 |
| 打包 | PyInstaller（Windows 优先） | Briefcase | 成熟稳定 |

**翻译引擎抽象**：定义 `Translator` 接口，NLLB / Argos 各实现一份，配置可切换。

---

## 5. 系统架构

```
GUI (PySide6)
 └─ Application Service          任务编排 / 进度 / 取消
     └─ Pipeline Core
         ├─ MediaProbe / FFmpeg   抽音轨 → 16k mono
         ├─ AsrEngine             源语识别（词级/段级时间戳）
         ├─ Segmenter             按语言参数二次断句 + 时间轴规整
         ├─ Translator            NLLB / Argos（可插拔）
         ├─ SubtitleWriter        SRT / VTT / ASS，单语 / 双语
         └─ PostCheck             CPS / 行长 / 重叠 / 时长校验
     ├─ ModelRegistry            模型发现 / 校验 / 路径解析
     └─ Config / Logging / Cache
```

数据流：`文件 → 音轨 → 段列表(源语) → 分段规整 → 译文段 → 合并 → 字幕文件`

---

## 6. 目录结构

```
subextractor/
├─ docs/
│  ├─ design-outline.md
│  └─ iteration-plan.md
├─ src/subextractor/
│  ├─ core/
│  │  ├─ media.py        # FFmpeg 抽音轨
│  │  ├─ asr.py          # faster-whisper 封装
│  │  ├─ segment.py      # 断句 / 时间轴规整
│  │  ├─ translate.py    # 翻译引擎抽象 + NLLB / Argos
│  │  ├─ subtitle.py     # 字幕数据模型
│  │  ├─ writer.py       # SRT / VTT / ASS 写入
│  │  ├─ postcheck.py    # 输出质量校验
│  │  └─ pipeline.py     # 端到端编排
│  ├─ models/registry.py
│  ├─ config/            # 语言参数表 + 默认配置
│  ├─ ui/                # PySide6 界面
│  └─ utils/
├─ models/               # 用户放置的离线模型（不进包）
│  ├─ asr/
│  ├─ mt/
│  └─ vad/
├─ tests/
└─ pyproject.toml
```

---

## 7. 数据模型

```python
@dataclass
class Word:
    text: str
    start: float      # 秒
    end: float

@dataclass
class Segment:            # 一条字幕（一个 cue）
    index: int
    text: str             # 源语
    translation: str | None
    start: float
    end: float
    words: list[Word]     # 可选，用于二次切分

@dataclass
class SubtitleDoc:
    source_lang: str
    target_lang: str | None
    segments: list[Segment]
```

---

## 8. 断句与时间轴规则

输入为 ASR 的词级/段级时间戳，二次切分保证字幕可读：

- 单 cue ≤ 2 行；每行 ≤ 该语言字符阈值（见 §3）
- cue 时长：下限 0.8s，上限 7s
- 相邻 cue **不重叠**（重叠自动修剪至此前一条的 end）
- 过短 cue（< 0.8s）合并到相邻 cue
- 换行点优先落在：句读标点 > 连词/助词 > 词边界（en/fr）
- ja/ko 专项：不得以助词、助动词、格助词开头断行；日文按文节切

所有阈值来自 `config/languages.yaml`，便于实测调参。

---

## 9. 输出规范

对每个输入文件 `name.ext` 产出：

| 文件 | 内容 |
|---|---|
| `name.orig.srt` | 源语字幕 |
| `name.<lang>.srt` | 目标语字幕 |
| `name.bilingual.<lang>.srt` | 双语字幕：第 1 行源语，第 2 行译文 |
| `name.bilingual.<lang>.ass` | 可选，可为东亚/拉丁字符分别设定样式 |

- 编码 UTF-8（可选带 BOM）
- SRT 时间戳格式 `HH:MM:SS,mmm`
- 兼容目标：PotPlayer / VLC / mpv 直接外挂加载

---

## 10. 离线模型管理

- 目录约定：`models/asr/`、`models/mt/`、`models/vad/`
- 首次需在联网机器下载后拷入，附 `models/README` 与 `checksums.json`
- 启动时**只做本地扫描**：缺失时给出明确提示与放置指引，绝不联网回退
- 建议模型档位：ASR = faster-whisper `small`/`medium`/`large-v3`；MT = NLLB-200-600M int8；VAD = Silero

---

## 11. 性能与硬件

- **CPU 可用**：faster-whisper base/small + int8，NLLB int8 亦可跑，慢但可接受
- **GPU 推荐**：CUDA 12 / cuDNN 9，large-v3 约需 6–8GB 显存
- 长视频：按 VAD 静音切分处理，支持断点续跑

---

## 12. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| ja/ko 无空格，断句难 | 字幕可读性差 | 专用断句规则 + 人工抽检评测集 |
| Whisper 中日/日英混淆 | 识别错误 | 提供"强制指定源语"选项 |
| NLLB 术语不准 | 译文生硬 | 术语表 + 后处理替换 |
| 模型体积/显存 | 部署门槛 | int8 量化 + 多档模型可选 |
| 播放器兼容差异 | 外挂显示异常 | 以 SRT 为主输出，ASS 为可选 |
| 打包体积大 | 分发不便 | 模型外置，不进安装包 |

---

## 13. 许可合规（自用场景）

- faster-whisper / WhisperX / stable-ts / Argos / pysrt / PySide6：MIT 或 Apache-2.0 等宽松许可
- **NLLB-200 权重为 CC-BY-NC-4.0（非商用）**，自用无碍，不得商用分发
- Whisper 权重为 MIT

---

## 14. 参考轮子（借鉴而非照搬）

| 项目 | 可借鉴之处 |
|---|---|
| pyVideoTrans | 中文桌面完整流程、双语字幕输出、多 STT 后端 |
| SmartSub（妙幕） | 一站式"字幕→翻译(双语)→校对"体验 |
| WhisperSubTranslate | Electron + whisper.cpp 的轻量本地实现 |
| VideoLingo | WhisperX 词级对齐 + 字幕切分规范（Netflix 风格） |
| Vibe / Buzz | 离线桌面转录 App 的工程与交互做法 |
