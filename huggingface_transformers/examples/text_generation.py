"""
文本生成示例：使用GPT模型进行文本生成
此示例展示如何使用Hugging Face Transformers进行文本生成任务
"""

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    TextGenerationPipeline,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import torch
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextGenerator:
    """
    文本生成器类，封装了不同的文本生成方法
    """
    def __init__(self, model_name: str = "gpt2"):
        """
        初始化文本生成器
        
        Args:
            model_name: 要使用的模型名称，默认为'gpt2'
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        # 初始化tokenizer和model
        logger.info(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # 如果需要，设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 创建pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def generate_simple(self, prompt: str, max_length: int = 100) -> str:
        """
        使用简单的pipeline方法生成文本

        Args:
            prompt: 输入提示文本
            max_length: 生成文本的最大长度

        Returns:
            生成的文本
        """
        logger.info("使用pipeline方法生成文本")
        result = self.pipeline(
            prompt,
            max_length=max_length,
            num_return_sequences=1
        )[0]["generated_text"]

        return result

    def generate_advanced(
        self,
        prompt: str,
        max_length: int = 100,
        num_sequences: int = 1,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = True
    ) -> List[str]:
        """
        使用高级参数生成文本

        Args:
            prompt: 输入提示文本
            max_length: 生成文本的最大长度
            num_sequences: 生成的序列数量
            temperature: 生成的随机性（越高越随机）
            top_k: 在每一步保留的最可能的token数量
            top_p: 累积概率的阈值
            repetition_penalty: 重复惩罚系数
            do_sample: 是否使用采样（如果为False，则使用贪婪解码）

        Returns:
            生成的文本列表
        """
        logger.info("使用高级参数生成文本")

        # 编码输入文本
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成文本
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 解码生成的文本
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return generated_texts

    def interactive_generation(self):
        """
        交互式文本生成会话
        """
        logger.info("开始交互式生成会话（输入'quit'退出）")

        while True:
            # 获取用户输入
            prompt = input("\n请输入提示文本: ")

            if prompt.lower() == 'quit':
                break

            # 获取生成参数
            try:
                num_sequences = int(input("生成多少个不同的序列？(1-5): "))
                num_sequences = max(1, min(5, num_sequences))

                temperature = float(input("设置temperature (0.1-2.0，越高越随机): "))
                temperature = max(0.1, min(2.0, temperature))

                max_length = int(input("最大生成长度 (10-500): "))
                max_length = max(10, min(500, max_length))
            except ValueError:
                logger.warning("输入无效，使用默认值")
                num_sequences = 1
                temperature = 0.7
                max_length = 100

            # 生成文本
            generated_texts = self.generate_advanced(
                prompt,
                max_length=max_length,
                num_sequences=num_sequences,
                temperature=temperature
            )

            # 打印结果
            print("\n生成的文本:")
            for i, text in enumerate(generated_texts, 1):
                print(f"\n--- 序列 {i} ---")
                print(text)

def main():
    """
    主函数：展示不同的文本生成方法
    """
    # 创建生成器实例
    generator = TextGenerator()

    # 1. 简单的pipeline方法
    print("\n=== 使用简单的pipeline方法 ===")
    prompt = "Once upon a time in a magical forest,"
    generated_text = generator.generate_simple(prompt)
    print(f"\n提示: {prompt}")
    print(f"生成的文本: {generated_text}")

    # 2. 使用高级参数生成多个不同的序列
    print("\n=== 使用高级参数生成多个序列 ===")
    prompts = [
        "The artificial intelligence revolution will",
        "In the next decade, space exploration will",
        "The future of renewable energy depends on"
    ]

    for prompt in prompts:
        print(f"\n提示: {prompt}")
        generated_texts = generator.generate_advanced(
            prompt,
            num_sequences=2,
            temperature=0.8,
            max_length=100
        )

        for i, text in enumerate(generated_texts, 1):
            print(f"\n变体 {i}:")
            print(text)

    # 3. 交互式生成（可选）
    user_input = input("\n是否要启动交互式生成会话？(y/n): ")
    if user_input.lower() == 'y':
        generator.interactive_generation()

if __name__ == "__main__":
    main()
