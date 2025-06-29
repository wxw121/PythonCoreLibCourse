# Hugging Face Transformers 模型微调指南

## 什么是微调？

微调(Fine-tuning)是一种迁移学习技术，它利用预训练模型作为起点，然后通过在特定任务的数据集上进一步训练来调整模型参数。这种方法比从头训练模型更高效，因为预训练模型已经学习了语言的一般特征，只需要适应特定任务的特点。

## 微调的优势

1. **更少的数据需求**：微调通常只需要相对较小的数据集，因为模型已经从大量数据中学习了基本的语言理解能力。
2. **更快的训练速度**：从预训练模型开始，训练时间大大缩短。
3. **更好的性能**：对于大多数任务，微调预训练模型比从头训练模型性能更好。
4. **资源效率**：微调需要的计算资源比从头训练少得多。

## 微调前的准备

### 1. 确定任务类型

首先，明确你要解决的问题类型：
- 文本分类（如情感分析、主题分类）
- 序列标注（如命名实体识别、词性标注）
- 问答
- 文本生成
- 摘要
- 翻译
- 等等

### 2. 选择合适的预训练模型

根据任务类型和语言选择合适的预训练模型：
- BERT/RoBERTa：适合理解任务（分类、NER等）
- GPT-2/GPT-3：适合生成任务
- T5/BART：适合序列到序列任务（翻译、摘要等）
- DistilBERT/TinyBERT：轻量级模型，适合资源受限的环境

### 3. 准备数据集

准备适合你任务的数据集，通常包括：
- 训练集：用于模型学习
- 验证集：用于调整超参数和早停
- 测试集：用于最终评估

## 使用Hugging Face的Trainer API进行微调

Hugging Face提供了`Trainer`和`TrainingArguments`类，简化了模型微调过程。

### 基本微调流程

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 加载数据集
dataset = load_dataset("glue", "sst2")  # 情感分析数据集

# 2. 加载预训练模型和分词器
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. 定义评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 5. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",              # 输出目录
    num_train_epochs=3,                  # 训练轮数
    per_device_train_batch_size=16,      # 每个设备的训练批量大小
    per_device_eval_batch_size=64,       # 每个设备的评估批量大小
    warmup_steps=500,                    # 预热步数
    weight_decay=0.01,                   # 权重衰减
    logging_dir="./logs",                # 日志目录
    logging_steps=10,                    # 日志记录间隔
    evaluation_strategy="epoch",         # 每个epoch评估一次
    save_strategy="epoch",               # 每个epoch保存一次
    load_best_model_at_end=True,         # 训练结束时加载最佳模型
    metric_for_best_model="accuracy",    # 用于选择最佳模型的指标
)

# 6. 初始化Trainer
trainer = Trainer(
    model=model,                         # 预训练模型
    args=training_args,                  # 训练参数
    train_dataset=tokenized_datasets["train"],  # 训练数据集
    eval_dataset=tokenized_datasets["validation"],  # 验证数据集
    compute_metrics=compute_metrics,     # 评估指标
)

# 7. 开始微调
trainer.train()

# 8. 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 9. 保存模型
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
```

## 不同任务的微调示例

### 1. 文本分类（情感分析）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 加载数据集
dataset = load_dataset("imdb")  # IMDB电影评论数据集

# 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 定义评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results_imdb",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_imdb",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# 开始微调
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
```

### 2. 命名实体识别(NER)

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

# 加载数据集
dataset = load_dataset("conll2003")  # CoNLL-2003数据集

# 加载预训练模型和分词器
model_name = "bert-base-cased"  # 注意：对于NER，通常使用cased模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 获取标签列表
label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)

# 加载模型
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# 数据预处理
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
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

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 定义评估指标
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # 移除忽略的索引（-100）
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results_ner",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_ner",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# 开始微调
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
```

### 3. 问答系统

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import torch
from tqdm.auto import tqdm

# 加载数据集
dataset = load_dataset("squad")  # SQuAD问答数据集

# 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 数据预处理
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    
    # 分词
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
    
    # 映射原始示例索引到特征
    sample_map = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")
    
    # 标签
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
        
        # 找到问题和上下文的分隔点
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1
        
        # 如果答案不在上下文中，标记为CLS位置
        if start_char < 0 or end_char > len(contexts[sample_idx]) or contexts[sample_idx][start_char:end_char] != answer["text"][0]:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # 否则，找到答案的开始和结束位置
            token_start_index = context_start
            while token_start_index < len(offset) and offset[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1
            
            token_end_index = context_end
            while token_end_index >= 0 and offset[token_end_index][1] >= end_char:
                token_end_index -= 1
            token_end_index += 1
            
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# 处理数据集
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results_qa",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_qa",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 开始微调
trainer.train()

# 保存模型
model.save_pretrained("./fine_tuned_qa_model")
tokenizer.save_pretrained("./fine_tuned_qa_model")
```

### 4. 文本生成（语言模型微调）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# 加载预训练模型和分词器
model_name = "gpt2"  # 使用GPT-2模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 数据整理函数
def group_texts(examples):
    # 将所有文本连接起来
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # 将连接的文本分成块
    block_size = 128
    total_length = (total_length // block_size) * block_size
    
    # 创建新的示例
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # 创建标签（与输入相同，用于自回归训练）
    result["labels"] = result["input_ids"].copy()
    
    return result

# 应用分组函数
lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results_lm",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_lm",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# 开始微调
trainer.train()

# 保存模型
model.save_pretrained("./fine_tuned_lm_model")
tokenizer.save_pretrained("./fine_tuned_lm_model")

# 使用微调后的模型生成文本
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 示例
prompt = "In this tutorial, we will learn how to"
generated_text = generate_text(prompt)
print(f"Generated text: {generated_text}")
```

### 5. 文本摘要

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer

# 下载nltk资源
nltk.download('punkt')

# 加载数据集
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 加载预训练模型和分词器
model_name = "t5-small"  # 使用T5模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 数据预处理
def preprocess_function(examples):
    # 添加前缀"summarize: "，因为T5是一个多任务模型
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    # 设置摘要作为标签
    labels = tokenizer(examples["highlights"], max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 处理数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 定义评估指标
def compute_metrics(pred):
    predictions, labels = pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 替换-100为tokenizer.pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 后处理：将生成的摘要和参考摘要分成句子
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
    # 计算ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1 = rouge2 = rougeL = 0.0
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(pred, label)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
    
    rouge1 /= len(decoded_preds)
    rouge2 /= len(decoded_preds)
    rougeL /= len(decoded_preds)
    
    return {
        'rouge1': rouge1,
        'rouge2': rouge2,
        'rougeL': rougeL,
    }

# 设置训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_summarization",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs_summarization",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True,
    generation_max_length=128,
)

# 初始化Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# 开始微调
trainer.train()

# 保存模型
model.save_pretrained("./fine_tuned_summarization_model")
tokenizer.save_pretrained("./fine_tuned_summarization_model")
```

## 高级微调技术

### 1. 梯度累积

当GPU内存有限时，可以使用梯度累积来模拟更大的批量大小：

```python
# 在TrainingArguments中设置
training_args = TrainingArguments(
    # ...其他参数...
    gradient_accumulation_steps=4,  # 累积4个批次的梯度再更新
)
```

### 2. 混合精度训练

使用混合精度训练可以加速训练并减少内存使用：

```python
# 在TrainingArguments中设置
training_args = TrainingArguments(
    # ...其他参数...
    fp16=True,  # 启用混合精度训练
)
```

### 3. 学习率调度

合适的学习率调度可以提高模型性能：

```python
# 在TrainingArguments中设置
training_args = TrainingArguments(
    # ...其他参数...
    learning_rate=5e-5,
    warmup_steps=500,
    lr_scheduler_type="linear",  # 线性学习率调度
)
```

### 4. 早停法

当验证集性能不再提高时停止训练，避免过拟合：

```python
# 在TrainingArguments中设置
training_args = TrainingArguments(
    # ...其他参数...
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # 或其他适合你任务的指标
    early_stopping_patience=3,  # 3个评估周期没有改善就停止
)
```

### 5. 参数高效微调 (PEFT)

对于大型模型，可以使用参数高效微调方法，如LoRA（低秩适应）：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# 加载预训练模型
model_name = "bert-large-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # LoRA的秩
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]  # 要微调的模块
)

# 获取PEFT模型
model = get_peft_model(model, peft_config)

# 现在可以像正常微调一样使用Trainer
# ...
```

## 微调后的模型评估与部署

### 评估模型

```python
# 使用Trainer评估
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 手动评估
from transformers import pipeline

# 加载微调后的模型
model_path = "./fine_tuned_model"
classifier = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

# 评估样本
samples = ["This movie was fantastic!", "I didn't enjoy this book at all."]
results = classifier(samples)
print(results)
```

### 部署模型

#### 1. 使用Hugging Face模型中心

```python
from huggingface_hub import notebook_login
import os

# 登录Hugging Face
notebook_login()

# 推送模型到Hub
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

#### 2. 使用TorchServe部署

```python
# 安装TorchServe
# pip install torchserve torch-model-archiver

# 创建模型存档
# torch-model-archiver --model-name bert_model \
#                      --version 1.0 \
#                      --serialized-file path/to/model.pt \
#                      --handler transformers_handler.py \
#                      --extra-files "path/to/tokenizer/,path/to/config.json"

# 启动TorchServe
# torchserve --start --model-store model_store --models bert=bert_model.mar
```

#### 3. 使用FastAPI创建API

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# 加载模型和分词器
model_path = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: TextInput):
    # 准备输入
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)
    
    # 进行推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 处理输出
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_id = predictions.argmax().item()
    label = model.config.id2label[label_id]
    score = predictions[0][label_id].item()
    
    return {"label": label, "score": score}

# 启动服务器
# uvicorn app:app --host 0.0.0.0 --port 8000
```

## 常见问题与解决方案

### 1. 过拟合

当模型在训练集上表现良好但在验证集上表现不佳时，可能发生过拟合。

解决方案：
- 使用更多的训练数据
- 应用正则化技术（如权重衰减、Dropout）
- 减小模型大小
- 使用早停法
- 数据增强

### 2. 欠拟合

当模型在训练集和验证集上都表现不佳时，可能发生欠拟合。

解决方案：
- 使用更大、更复杂的模型
- 增加训练轮数
- 调整学习率
- 减少正则化强度

### 3. 内存不足

当处理大型模型或大量数据时，可能会遇到内存不足的问题。

解决方案：
- 减小批量大小
- 使用梯度累积
- 使用混合精度训练
- 使用较小的模型
- 使用参数高效微调方法（如LoRA）

### 4. 训练速度慢

解决方案：
- 使用GPU或TPU加速
- 使用混合精度训练
- 减小模型大小
- 使用更高效的优化器
- 使用更小的数据集进行初步实验

## 下一步

现在你已经了解了如何微调Transformers模型，可以继续学习：

1. 探索 `examples/` 目录中的实际应用示例
2. 学习 `advanced_topics.md` 了解更高级的技巧和最佳实践
3. 尝试将微调后的模型集成到实际应用中