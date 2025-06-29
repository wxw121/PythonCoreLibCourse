"""
命名实体识别(NER)示例：使用Transformers进行序列标注
此示例展示如何使用Hugging Face Transformers进行命名实体识别任务
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, classification_report

def simple_approach():
    """
    使用pipeline API的简单方法
    """
    print("=== 使用Pipeline API的简单方法 ===")
    
    # 创建NER pipeline
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")
    
    # 准备一些测试文本
    texts = [
        "My name is Sarah and I work at Microsoft in Seattle.",
        "The Eiffel Tower is located in Paris, France.",
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."
    ]

    # 进行预测
    for text in texts:
        print(f"\n文本: {text}")
        results = ner_pipeline(text)

        print("识别的实体:")
        for entity in results:
            print(f"- {entity['word']} ({entity['entity_group']}): 置信度 {entity['score']:.4f}")

def advanced_approach():
    """
    使用完整训练流程的高级方法
    """
    print("\n=== 使用完整训练流程的高级方法 ===")

    # 加载数据集
    dataset = load_dataset("conll2003")

    # 加载预训练模型和分词器
    model_name = "bert-base-cased"  # 注意：对于NER，通常使用cased模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 获取标签列表
    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    # 加载模型
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # 数据预处理
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True
        )
        labels = []

        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # 特殊标记的标签为-100
                if word_idx is None:
                    label_ids.append(-100)
                # 对于第一个标记，使用对应的标签
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # 对于同一个词的其他标记，也使用-100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # 处理训练集的一小部分（为了演示）
    small_train_dataset = dataset["train"].select(range(1000))
    small_eval_dataset = dataset["validation"].select(range(100))

    tokenized_train_dataset = small_train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=small_train_dataset.column_names
    )

    tokenized_eval_dataset = small_eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=small_eval_dataset.column_names
    )

    # 准备数据加载器
    tokenized_train_dataset.set_format("torch")
    tokenized_eval_dataset.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_train_dataset, shuffle=True, batch_size=8
    )

    eval_dataloader = DataLoader(
        tokenized_eval_dataset, batch_size=8
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

    predictions = []
    true_labels = []

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions_batch = torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        for i in range(len(labels)):
            pred = predictions_batch[i].cpu().numpy()
            label = labels[i].cpu().numpy()

            # 过滤掉特殊标记（-100）
            true_prediction = [id2label[p] for p, l in zip(pred, label) if l != -100]
            true_label = [id2label[l] for l in label if l != -100]

            predictions.append(true_prediction)
            true_labels.append(true_label)

    # 计算并打印评估指标
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    print(f"\n准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(true_labels, predictions))

def main():
    """
    主函数：运行两种方法的示例
    """
    # 运行简单方法
    simple_approach()

    # 运行高级方法（可选，因为训练可能需要较长时间）
    # 如果只想看简单的pipeline示例，可以注释掉下面这行
    # advanced_approach()

if __name__ == "__main__":
    main()
