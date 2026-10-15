"""管线核心层（设计大纲 §5）。

模块职责：
- media     音轨抽取与媒体探测
- asr       源语识别（faster-whisper）
- segment   按语言参数二次断句、时间轴规整
- translate 翻译引擎抽象（NLLB / Argos）
- subtitle  字幕数据模型
- writer    字幕文件写入（SRT / VTT / ASS）
- postcheck 输出质量校验
- pipeline  端到端编排
"""
