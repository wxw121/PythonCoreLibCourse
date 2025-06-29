"""
文本分类示例：使用BERT模型进行情感分析
此示例展示如何使用Hugging Face Transformers进行文本分类任务
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def simple_approach():
    """
    使用pipeline API的简单方法
    """
    print("=== 使用Pipeline API的简单方法 ===")

    # 创建情感分析pipeline
    classifier = pipeline("sentiment-analysis")

    # 准备一些测试文本
    texts = [
        "我真的很喜欢这部电影，情节很吸引人！",
        "这是我看过最糟糕的一部电影。",
        "这部电影还行，但不是特别出色。"
    ]

    # 进行预测
    results = classifier(texts)

    # 打印结果
    for text, result in zip(texts, results):
        print(f"\n文本: {text}")
        print(f"情感: {result['label']}")
        print(f"置信度: {result['score']:.4f}")

def advanced_approach():
    """
    使用完整训练流程的高级方法
    """
    print("\n=== 使用完整训练流程的高级方法 ===")

    # 加载数据集
    dataset = load_dataset("imdb", split="train[:1000]")  # 为了演示，只使用1000个样本

    # 加载预训练模型和分词器
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 数据预处理
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 准备数据加载器
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=8
    )

    # 准备训练
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # 使用GPU（如果可用）
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # 训练循环
    print("\n开始训练...")
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        print(f"Epoch {epoch + 1}/{num_epochs} 完成")

    # 评估
    print("\n进行评估...")
    model.eval()
    test_dataset = load_dataset("imdb", split="test[:100]")  # 为了演示，只使用100个样本

    predictions = []
    true_labels = []

    for text, label in zip(test_dataset["text"], test_dataset["label"]):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        pred = outputs.logits.argmax(-1).item()
        predictions.append(pred)
        true_labels.append(label)

    # 计算并打印评估指标
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\n准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(true_labels, predictions))

def main():
    """
    主函数：运行两种方法的示例
    """
    # 运行简单方法
    simple_approach()

    # 运行高级方法
    advanced_approach()

if __name__ == "__main__":
    main()
