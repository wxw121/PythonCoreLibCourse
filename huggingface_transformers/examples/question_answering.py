"""
问答系统示例：使用Transformers进行问答任务
此示例展示如何使用Hugging Face Transformers构建问答系统
"""

from transformers import (
    AutoModelForQuestionAnswering, 
    AutoTokenizer, 
    pipeline,
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
import torch
import numpy as np
from tqdm.auto import tqdm
import logging
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionAnsweringSystem:
    """
    问答系统类，封装了不同的问答方法
    """
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        初始化问答系统
        
        Args:
            model_name: 要使用的模型名称，默认为'distilbert-base-cased-distilled-squad'
                        （这是一个在SQuAD数据集上微调过的模型）
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        # 初始化tokenizer和model
        logger.info(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device)
        
        # 创建pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def answer_question_simple(self, question: str, context: str) -> Dict[str, Any]:
        """
        使用简单的pipeline方法回答问题
        
        Args:
            question: 问题文本
            context: 上下文文本（包含答案的段落）
            
        Returns:
            包含答案、分数和起始/结束位置的字典
        """
        logger.info("使用pipeline方法回答问题")
        result = self.qa_pipeline(question=question, context=context)
        return result
    
    def answer_question_advanced(
        self, 
        question: str, 
        context: str,
        max_seq_length: int = 384,
        doc_stride: int = 128
    ) -> Dict[str, Any]:
        """
        使用高级方法回答问题，手动处理长文本和重叠窗口
        
        Args:
            question: 问题文本
            context: 上下文文本（可能很长）
            max_seq_length: 最大序列长度
            doc_stride: 文档步长（窗口重叠大小）
            
        Returns:
            包含答案、分数和起始/结束位置的字典
        """
        logger.info("使用高级方法回答问题")
        
        # 对问题和上下文进行编码
        inputs = self.tokenizer(
            question,
            context,
            max_length=max_seq_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )

        # 获取输入ID和偏移映射
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offset_mapping = inputs["offset_mapping"]
        overflow_to_sample_mapping = inputs["overflow_to_sample_mapping"]

        # 进行预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        # 处理所有特征的预测
        all_predictions = []

        for i in range(len(input_ids)):
            # 获取此特征的起始和结束位置
            start_logit = start_logits[i].cpu().numpy()
            end_logit = end_logits[i].cpu().numpy()
            offset_map = offset_mapping[i].cpu().numpy()

            # 找到最佳的起始和结束位置
            # 忽略[CLS]标记、[SEP]标记和填充标记
            cls_index = input_ids[i].tolist().index(self.tokenizer.cls_token_id)

            # 找到最佳的起始和结束位置
            start_indexes = np.argsort(start_logit)[-20:][::-1]  # 取前20个最可能的起始位置
            end_indexes = np.argsort(end_logit)[-20:][::-1]  # 取前20个最可能的结束位置

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 跳过不合理的答案
                    if (
                        start_index >= len(offset_map)
                        or end_index >= len(offset_map)
                        or offset_map[start_index] is None
                        or offset_map[end_index] is None
                        or start_index <= cls_index  # 答案不能在[CLS]之前
                        or end_index <= cls_index  # 答案不能在[CLS]之前
                        or end_index < start_index  # 结束不能在开始之前
                        or end_index - start_index + 1 > 30  # 答案不能太长
                    ):
                        continue

                    # 计算分数
                    score = start_logit[start_index] + end_logit[end_index]

                    # 获取原始文本中的字符偏移
                    char_start_index = offset_map[start_index][0]
                    char_end_index = offset_map[end_index][1]

                    # 提取答案文本
                    answer = context[char_start_index:char_end_index]

                    all_predictions.append({
                        "text": answer,
                        "score": float(score),
                        "start": int(char_start_index),
                        "end": int(char_end_index)
                    })

        # 如果没有找到答案
        if not all_predictions:
            return {"text": "", "score": 0.0, "start": 0, "end": 0}

        # 按分数排序并返回最佳答案
        all_predictions.sort(key=lambda x: x["score"], reverse=True)
        return all_predictions[0]

    def interactive_qa(self):
        """
        交互式问答会话
        """
        logger.info("开始交互式问答会话（输入'quit'退出）")

        # 默认上下文
        default_context = """
        Hugging Face是一家人工智能公司，成立于2016年，总部位于纽约和巴黎。
        该公司开发了Transformers库，这是一个用于自然语言处理的开源库，
        支持PyTorch、TensorFlow和JAX等深度学习框架。
        Hugging Face还维护着一个模型中心，称为Hugging Face Hub，
        其中包含了数千个预训练模型，可以用于各种NLP任务，如文本分类、
        命名实体识别、问答、摘要生成等。
        2021年，Hugging Face的估值达到了20亿美元，成为AI领域的独角兽公司。
        """

        current_context = default_context

        while True:
            print("\n当前上下文:")
            print("-" * 50)
            print(current_context)
            print("-" * 50)

            # 获取用户输入
            question = input("\n请输入问题（输入'quit'退出，输入'context'更改上下文）: ")

            if question.lower() == 'quit':
                break

            if question.lower() == 'context':
                new_context = input("\n请输入新的上下文文本: ")
                if new_context:
                    current_context = new_context
                continue

            # 回答问题
            answer = self.answer_question_simple(question, current_context)

            # 打印结果
            print("\n回答:", answer["answer"])
            print(f"置信度: {answer['score']:.4f}")
            print(f"起始位置: {answer['start']}")
            print(f"结束位置: {answer['end']}")

def fine_tune_qa_model(output_dir: str = "./fine_tuned_qa_model"):
    """
    在SQuAD数据集上微调问答模型

    Args:
        output_dir: 保存微调模型的目录
    """
    logger.info("开始微调问答模型")

    # 加载数据集
    logger.info("加载SQuAD数据集")
    dataset = load_dataset("squad")

    # 加载预训练模型和分词器
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # 数据预处理函数
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]

        # 编码输入
        inputs = tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # 准备答案
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # 找到CLS标记的位置
            sequence_ids = inputs.sequence_ids(i)
            cls_index = sequence_ids.index(0)

            # 找到包含答案的文本部分
            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1

            # 如果答案不在上下文中，标记为不可能
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # 否则，找到答案的起始和结束标记位置
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    # 处理数据集
    logger.info("预处理数据集")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # 设置训练参数
    logger.info("设置训练参数")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # 为了演示，只训练1个epoch
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs_qa",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 初始化Trainer
    logger.info("初始化Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # 开始微调
    logger.info("开始微调")
    trainer.train()

    # 保存模型
    logger.info(f"保存模型到 {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("微调完成")
    return output_dir

def main():
    """
    主函数：展示不同的问答方法
    """
    # 创建问答系统实例
    qa_system = QuestionAnsweringSystem()

    # 1. 简单的pipeline方法
    print("\n=== 使用简单的pipeline方法 ===")
    context = """
    Hugging Face是一家人工智能公司，成立于2016年，总部位于纽约和巴黎。
    该公司开发了Transformers库，这是一个用于自然语言处理的开源库，
    支持PyTorch、TensorFlow和JAX等深度学习框架。
    """

    questions = [
        "Hugging Face是什么时候成立的？",
        "Hugging Face的总部在哪里？",
        "Transformers库支持哪些深度学习框架？"
    ]

    for question in questions:
        answer = qa_system.answer_question_simple(question, context)
        print(f"\n问题: {question}")
        print(f"回答: {answer['answer']}")
        print(f"置信度: {answer['score']:.4f}")

    # 2. 使用高级方法处理长