# -*- coding: utf-8 -*-
# 把离线模型放到本目录（模型权重不随安装包分发）：
#   asr\faster-whisper-small\            识别模型（必需，至少一个档位）
#   asr\faster-whisper-large-v3\         高质量识别档（可选，建议配 GPU）
#   mt\nllb-200-distilled-600M-ct2-int8\ 翻译模型（必需）
#
# 在联网电脑上获取后拷贝到本机：
#   subextractor-cli.exe fetch-model --size small
#   subextractor-cli.exe fetch-model --kind mt
#
# 也可以启动 SubExtractor.exe 后在「帮助 - 模型准备向导」里选择已有模型目录。
