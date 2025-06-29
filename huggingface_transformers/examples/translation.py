"""
机器翻译示例：使用Transformers进行多语言翻译
此示例展示如何使用Hugging Face Transformers进行机器翻译任务
"""

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    MBartForConditionalGeneration,
    MBart50Tokenizer
)
import sentencepiece
import torch
from typing import List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Translator:
    """
    翻译器类，支持多种语言之间的翻译
    """
    def __init__(self, src_lang: str = "zh_CN", tgt_lang: str = "en_XX"):
        """
        初始化翻译器

        Args:
            src_lang: 源语言代码，默认为中文
            tgt_lang: 目标语言代码，默认为英文
        """
        self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")

        # 初始化tokenizer和model
        logger.info(f"加载模型: {self.model_name}")
        self.tokenizer = MBart50Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.src_lang = self.src_lang
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

        # 创建翻译pipeline
        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def translate_simple(self, text: str) -> str:
        """
        使用简单的pipeline方法进行翻译

        Args:
            text: 要翻译的文本

        Returns:
            翻译后的文本
        """
        logger.info("使用pipeline方法进行翻译")
        result = self.translator(text, max_length=512)
        return result[0]["translation_text"]

    def translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        批量翻译多个文本

        Args:
            texts: 要翻译的文本列表
            batch_size: 批处理大小

        Returns:
            翻译后的文本列表
        """
        logger.info(f"批量翻译 {len(texts)} 个文本，批大小: {batch_size}")
        results = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # 编码
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

            # 翻译
            with torch.no_grad():
                translated_ids = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang]
                )

            # 解码
            batch_results = self.tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            results.extend(batch_results)

        return results

    def translate_advanced(
        self,
        text: str,
        num_beams: int = 5,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> List[str]:
        """
        使用高级参数进行翻译

        Args:
            text: 要翻译的文本
            num_beams: 束搜索的束数
            num_return_sequences: 返回的翻译数量
            temperature: 生成的随机性（越高越随机）
            do_sample: 是否使用采样（如果为False，则使用贪婪解码）

        Returns:
            翻译后的文本列表
        """
        logger.info("使用高级参数进行翻译")

        # 编码输入文本
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 生成翻译
        with torch.no_grad():
            translated_ids = self.model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                do_sample=do_sample,
                max_length=512,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang]
            )

        # 解码生成的文本
        translated_texts = self.tokenizer.batch_decode(translated_ids, skip_special_tokens=True)

        return translated_texts


class MultilingualTranslator:
    """
    多语言翻译器，支持多种语言对之间的翻译
    """
    def __init__(self):
        """
        初始化多语言翻译器
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")

        # 支持的语言对
        self.language_pairs = {
            "en-zh": "Helsinki-NLP/opus-mt-en-zh",
            "zh-en": "Helsinki-NLP/opus-mt-zh-en",
            "en-fr": "Helsinki-NLP/opus-mt-en-fr",
            "fr-en": "Helsinki-NLP/opus-mt-fr-en",
            "en-de": "Helsinki-NLP/opus-mt-en-de",
            "de-en": "Helsinki-NLP/opus-mt-de-en",
            "en-ru": "Helsinki-NLP/opus-mt-en-ru",
            "ru-en": "Helsinki-NLP/opus-mt-ru-en",
            "en-es": "Helsinki-NLP/opus-mt-en-es",
            "es-en": "Helsinki-NLP/opus-mt-es-en",
        }

        # 缓存已加载的模型
        self.models = {}
        self.tokenizers = {}
        self.target_lang_codes = {}  # 存储目标语言代码

    def _load_model_for_pair(self, src_lang: str, tgt_lang: str):
        """
        加载特定语言对的模型

        Args:
            src_lang: 源语言代码
            tgt_lang: 目标语言代码
        """
        lang_pair = f"{src_lang}-{tgt_lang}"

        if lang_pair not in self.language_pairs:
            raise ValueError(f"不支持的语言对: {lang_pair}")

        if lang_pair not in self.models:
            # 使用MBart模型替代Marian模型
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            logger.info(f"加载模型: {model_name}")

            # 将语言代码转换为MBart格式
            mbart_src_lang = self._convert_to_mbart_lang_code(src_lang)
            mbart_tgt_lang = self._convert_to_mbart_lang_code(tgt_lang)
            
            self.tokenizers[lang_pair] = MBart50Tokenizer.from_pretrained(model_name)
            self.tokenizers[lang_pair].src_lang = mbart_src_lang
            self.models[lang_pair] = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
            
            # 存储目标语言代码，用于生成时设置forced_bos_token_id
            self.target_lang_codes[lang_pair] = mbart_tgt_lang
            
    def _convert_to_mbart_lang_code(self, lang_code: str) -> str:
        """
        将简单的语言代码转换为MBart格式的语言代码
        
        Args:
            lang_code: 简单的语言代码，如'en', 'zh'等
            
        Returns:
            MBart格式的语言代码，如'en_XX', 'zh_CN'等
        """
        # 语言代码映射
        lang_map = {
            "en": "en_XX",
            "zh": "zh_CN",
            "fr": "fr_XX",
            "de": "de_DE",
            "ru": "ru_RU",
            "es": "es_XX"
        }
        
        if lang_code in lang_map:
            return lang_map[lang_code]
        elif "_" in lang_code:  # 如果已经是MBart格式
            return lang_code
        else:
            # 默认添加_XX后缀
            return f"{lang_code}_XX"

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        在指定的语言对之间翻译文本

        Args:
            text: 要翻译的文本
            src_lang: 源语言代码
            tgt_lang: 目标语言代码

        Returns:
            翻译后的文本
        """
        lang_pair = f"{src_lang}-{tgt_lang}"

        # 加载模型（如果尚未加载）
        self._load_model_for_pair(src_lang, tgt_lang)

        # 获取模型和分词器
        tokenizer = self.tokenizers[lang_pair]
        model = self.models[lang_pair]

        # 编码
        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        # 翻译，使用强制的目标语言token
        with torch.no_grad():
            translated_ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[self.target_lang_codes[lang_pair]],
                max_length=512
            )

        # 解码
        translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

        return translated_text

    def translate_auto(self, text: str, tgt_lang: str) -> str:
        """
        使用自动语言检测模型进行翻译

        Args:
            text: 要翻译的文本
            tgt_lang: 目标语言代码

        Returns:
            翻译后的文本
        """
        # 注意：这个方法需要一个多语言翻译模型，如M2M100或mBART
        # 这里使用M2M100作为示例
        model_name = "facebook/m2m100_418M"

        if "m2m" not in self.models:
            logger.info(f"加载多语言模型: {model_name}")
            self.tokenizers["m2m"] = AutoTokenizer.from_pretrained(model_name)
            self.models["m2m"] = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        tokenizer = self.tokenizers["m2m"]
        model = self.models["m2m"]

        # 设置目标语言
        tokenizer.src_lang = "zh"  # 这里假设源语言是中文，实际应用中可以使用语言检测
        tokenizer.tgt_lang = tgt_lang

        # 编码
        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        # 翻译
        with torch.no_grad():
            translated_ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
            )

        # 解码
        translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

        return translated_text


def interactive_translation():
    """
    交互式翻译会话
    """
    logger.info("开始交互式翻译会话（输入'quit'退出）")

    # 创建多语言翻译器
    translator = MultilingualTranslator()

    # 显示支持的语言对
    print("\n支持的语言对:")
    for lang_pair in translator.language_pairs:
        print(f"- {lang_pair}")

    while True:
        # 获取用户输入
        text = input("\n请输入要翻译的文本（输入'quit'退出）: ")

        if text.lower() == 'quit':
            break

        # 获取语言对
        src_lang = input("源语言代码（如'en', 'zh', 'fr'等）: ")
        tgt_lang = input("目标语言代码（如'en', 'zh', 'fr'等）: ")

        try:
            # 翻译文本
            translated_text = translator.translate(text, src_lang, tgt_lang)

            # 打印结果
            print("\n翻译结果:")
            print(translated_text)
        except ValueError as e:
            print(f"错误: {e}")


def main():
    """
    主函数：展示不同的翻译方法
    """
    # 1. 简单的中译英示例
    print("\n=== 中文到英文翻译示例 ===")
    zh_en_translator = Translator(src_lang="zh_CN", tgt_lang="en_XX")

    chinese_texts = [
        "人工智能正在改变我们的世界。",
        "机器学习是人工智能的一个子领域。",
        "深度学习模型在自然语言处理任务中表现出色。"
    ]

    for text in chinese_texts:
        translated = zh_en_translator.translate_simple(text)
        print(f"\n原文: {text}")
        print(f"译文: {translated}")

    # 2. 英译中示例
    print("\n=== 英文到中文翻译示例 ===")
    en_zh_translator = Translator(src_lang="en_XX", tgt_lang="zh_CN")

    english_texts = [
        "Artificial intelligence is changing our world.",
        "Machine learning is a subfield of artificial intelligence.",
        "Deep learning models perform excellently in natural language processing tasks."
    ]

    # 批量翻译
    translations = en_zh_translator.translate_batch(english_texts)

    for text, translated in zip(english_texts, translations):
        print(f"\n原文: {text}")
        print(f"译文: {translated}")

    # 3. 高级翻译参数示例
    print("\n=== 使用高级参数的翻译示例 ===")
    text = "The transformer architecture has revolutionized natural language processing."

    print(f"\n原文: {text}")
    print("\n使用不同的参数生成多个翻译变体:")

    # 使用束搜索
    translations = en_zh_translator.translate_advanced(
        text,
        num_beams=5,
        num_return_sequences=3,
        do_sample=False
    )

    for i, translation in enumerate(translations, 1):
        print(f"\n变体 {i} (束搜索): {translation}")

    # 使用采样
    translations = en_zh_translator.translate_advanced(
        text,
        num_beams=5,
        num_return_sequences=3,
        temperature=0.8,
        do_sample=True
    )

    for i, translation in enumerate(translations, 1):
        print(f"\n变体 {i} (温度采样): {translation}")

    # 4. 多语言翻译示例（可选）
    user_input = input("\n是否要启动交互式多语言翻译会话？(y/n): ")
    if user_input.lower() == 'y':
        interactive_translation()


if __name__ == "__main__":
    main()
