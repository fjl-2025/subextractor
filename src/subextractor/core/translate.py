"""翻译引擎抽象（设计大纲 §4）。

主力：NLLB-200-distilled-600M（CTranslate2 int8），覆盖 en / fr / ja / ko。
备选：Argos Translate（轻量，但对 ja/ko 覆盖较弱，多经英语中转）。
M2 实现具体引擎。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Translator(ABC):
    """翻译引擎接口，便于按配置切换实现。"""

    name: str = "base"

    @abstractmethod
    def translate(self, texts: list[str], src: str, dst: str) -> list[str]:
        """批量翻译，返回与输入等长的译文列表。"""

    def close(self) -> None:
        """释放资源（可选）。"""


class NllbTranslator(Translator):
    name = "nllb"

    def __init__(self, model_dir: str | Path | None = None, device: str = "cpu") -> None:
        self.model_dir = model_dir
        self.device = device

    def translate(self, texts: list[str], src: str, dst: str) -> list[str]:
        raise NotImplementedError("M2：NLLB 翻译待实现")


class ArgosTranslator(Translator):
    name = "argos"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = model_dir

    def translate(self, texts: list[str], src: str, dst: str) -> list[str]:
        raise NotImplementedError("M2：Argos 翻译待实现")


_ENGINES: dict[str, type[Translator]] = {
    NllbTranslator.name: NllbTranslator,
    ArgosTranslator.name: ArgosTranslator,
}


def create_translator(engine: str = "nllb", **kwargs: object) -> Translator:
    """按名称创建翻译引擎。"""
    if engine not in _ENGINES:
        supported = ", ".join(sorted(_ENGINES))
        raise ValueError(f"未知翻译引擎 {engine!r}，可选：{supported}")
    return _ENGINES[engine](**kwargs)
