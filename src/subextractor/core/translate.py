"""翻译引擎抽象与实现（设计大纲 §4）。

主力：NLLB-200-distilled-600M（CTranslate2 int8），覆盖 en / fr / ja / ko。
备选：Argos Translate（轻量，但对 ja/ko 覆盖较弱，多经英语中转；M2 暂不接入）。

NLLB 的用法（CTranslate2 官方范式）：
- 源语由 tokenizer 的 ``src_lang`` 决定：编码时会自动在句首插入源语标记；
- 目标语通过 ``target_prefix`` 强制指定，翻译结果首 token 即该标记，需剥掉。
语言用 FLORES-200 代码（``eng_Latn`` / ``fra_Latn`` / ``jpn_Jpan`` / ``kor_Hang`` …），
与 ASR（Whisper）输出的 ISO 639-1 代码（``en`` / ``fr`` / ``ja`` / ``ko``）需做映射。

模型只从本地目录加载（``local_files_only=True``），保证运行时离线。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from ..models import registry
from ..utils.logging import get_logger
from .errors import PipelineCancelled

logger = get_logger(__name__)

# ISO 639-1（Whisper 语言码）→ NLLB / FLORES-200 代码
NLLB_LANG_CODES: dict[str, str] = {
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "as": "asm_Beng",
    "az": "azj_Latn",
    "ba": "bak_Cyrl",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bo": "bod_Tibt",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fo": "fao_Latn",
    "fr": "fra_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "jv": "jav_Latn",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "ky": "kir_Cyrl",
    "lb": "ltz_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mg": "plt_Latn",
    "mi": "mri_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "nn": "nno_Latn",
    "no": "nob_Latn",
    "oc": "oci_Latn",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sa": "san_Deva",
    "sd": "snd_Arab",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "tk": "tuk_Latn",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "tt": "tat_Cyrl",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",
    "zh": "zho_Hans",
    "yue": "yue_Hant",
}


def to_nllb_code(lang: str) -> str:
    """ISO 639-1 → FLORES-200 代码；已是 FLORES 代码则原样返回。"""
    if "_" in lang:  # 已是 eng_Latn 之类的 FLORES 代码
        return lang
    code = NLLB_LANG_CODES.get(lang)
    if not code:
        raise ValueError(f"NLLB 不支持的语言代码 {lang!r}（无法映射到 FLORES-200）")
    return code


class Translator(ABC):
    """翻译引擎接口，便于按配置切换实现。"""

    name: str = "base"

    @abstractmethod
    def translate(
        self,
        texts: list[str],
        src: str,
        dst: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str]:
        """批量翻译，返回与输入等长的译文列表。

        ``should_cancel`` 每个批次结束检查一次，为真时抛 ``PipelineCancelled``。
        """

    def close(self) -> None:
        """释放资源（可选）。"""


class NllbTranslator(Translator):
    """NLLB-200-distilled-600M（CTranslate2 int8）实现。"""

    name = "nllb"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "auto",
        compute_type: str = registry.AUTO,
        batch_size: int = 16,
        beam_size: int = 4,
        max_decoding_length: int = 512,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_decoding_length = max_decoding_length
        self._translator = None
        self._tokenizer = None

    def _resolve_device(self) -> str:
        if self.device != registry.AUTO:
            return self.device
        return "cuda" if registry.cuda_device_count() > 0 else "cpu"

    def load(self) -> None:
        """惰性加载本地模型与 tokenizer（只执行一次）。"""
        if self._translator is not None:
            return

        import ctranslate2
        import transformers

        path = self.model_dir or registry.resolve_nllb_model()
        device = self._resolve_device()
        compute_type = self.compute_type
        if compute_type == registry.AUTO:
            # 模型权重本身是 int8；GPU 上用 int8_float16 加速，CPU 用 int8
            compute_type = "int8_float16" if device == "cuda" else "int8"

        logger.info("翻译引擎 nllb｜device=%s｜compute_type=%s", device, compute_type)
        self._translator = ctranslate2.Translator(
            str(path), device=device, compute_type=compute_type
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(path), local_files_only=True
        )

    def translate(
        self,
        texts: list[str],
        src: str,
        dst: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str]:
        self.load()

        src_code = to_nllb_code(src)
        dst_code = to_nllb_code(dst)
        self._tokenizer.src_lang = src_code

        translations: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            if should_cancel and should_cancel():
                raise PipelineCancelled("翻译已取消")
            batch = texts[start : start + self.batch_size]
            batch_tokens = [
                self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))
                for text in batch
            ]
            results = self._translator.translate_batch(
                batch_tokens,
                target_prefix=[[dst_code]] * len(batch),
                beam_size=self.beam_size,
                max_decoding_length=self.max_decoding_length,
            )
            for item in results:
                tokens = item.hypotheses[0]
                if tokens and tokens[0] == dst_code:  # 剥掉目标语标记
                    tokens = tokens[1:]
                ids = self._tokenizer.convert_tokens_to_ids(tokens)
                translations.append(self._tokenizer.decode(ids, skip_special_tokens=True))

        return translations

    def close(self) -> None:
        self._translator = None
        self._tokenizer = None


class ArgosTranslator(Translator):
    """Argos Translate 备选引擎（轻量）。

    依赖 argostranslate 及预置的语言包，M2 未接入（主力为 NLLB）。
    """

    name = "argos"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "auto",
        compute_type: str = registry.AUTO,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else None

    def translate(
        self,
        texts: list[str],
        src: str,
        dst: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[str]:
        raise NotImplementedError("Argos 备选引擎尚未接入（M2 起主力为 NLLB）")


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
