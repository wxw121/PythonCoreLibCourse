# Hugging Face Transformers 模型使用指南

## 模型概览

Hugging Face Transformers库提供了多种类型的预训练模型，适用于不同的任务和场景。本指南将详细介绍如何使用这些模型，以及它们各自的特点和适用场景。

## 模型类型

Transformers库中的模型主要可以分为以下几类：

### 1. 编码器模型 (Encoder Models)

编码器模型如BERT、RoBERTa和DistilBERT主要用于理解输入文本的含义，适合文本分类、命名实体识别等任务。

### 2. 解码器模型 (Decoder Models)

解码器模型如GPT、GPT-2和GPT-3主要用于生成文本，适合文本生成、故事创作等任务。

### 3. 编码器-解码器模型 (Encoder-Decoder Models)

编码器-解码器模型如T5、BART和Marian主要用于将一种形式的文本转换为另一种形式，适合翻译、摘要等任务。

### 4. 多模态模型 (Multimodal Models)

多模态模型如CLIP、ViT和BEiT可以处理多种类型的数据（如文本和图像），适合图像分类、图像-文本匹配等任务。

## 模型加载

Transformers库提供了多种方式来加载预训练模型：

### 使用Auto类

Auto类是加载模型最简单和推荐的方式，它会根据模型名称自动选择正确的模型类：

```python
from transformers import AutoModel, AutoTokenizer

# 加载模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

### 使用特定模型类

如果你知道要使用的确切模型类型，也可以直接使用特定的模型类：

```python
from transformers import BertModel, BertTokenizer

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

### 根据任务加载模型

Transformers提供了针对特定任务的模型类，如序列分类、问答等：

```python
from transformers import AutoModelForSequenceClassification

# 加载用于序列分类的模型
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
```

## 常用模型详解

### BERT (Bidirectional Encoder Representations from Transformers)

BERT是一个预训练的Transformer编码器模型，它通过双向上下文学习单词表示。

#### 特点
- 双向上下文理解
- 适合理解任务
- 多种语言版本可用

#### 使用示例

```python
from transformers import BertTokenizer, BertModel
import torch

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备输入
text = "Hello, my name is BERT."
encoded_input = tokenizer(text, return_tensors='pt')

# 获取模型输出
with torch.no_grad():
    output = model(**encoded_input)

# 获取[CLS]标记的最后隐藏状态，通常用作整个序列的表示
sequence_output = output.last_hidden_state
pooled_output = output.pooler_output  # [CLS]标记的表示，经过线性层和tanh激活

print(f"Sequence output shape: {sequence_output.shape}")  # [batch_size, sequence_length, hidden_size]
print(f"Pooled output shape: {pooled_output.shape}")      # [batch_size, hidden_size]
```

### GPT-2 (Generative Pre-trained Transformer 2)

GPT-2是一个预训练的Transformer解码器模型，专注于生成连贯的文本。

#### 特点
- 自回归模型（从左到右生成文本）
- 强大的文本生成能力
- 可以生成长篇、连贯的文本

#### 使用示例

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 加载预训练的GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 如果没有设置pad_token，使用eos_token代替
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 准备输入文本
text = "Once upon a time,"
input_ids = tokenizer.encode(text, return_tensors='pt')

# 生成文本
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
```

### RoBERTa (Robustly Optimized BERT Pretraining Approach)

RoBERTa是BERT的优化版本，通过更多的训练数据和更好的训练策略提高了性能。

#### 特点
- 比BERT训练时间更长，使用更多数据
- 移除了BERT的下一句预测任务
- 使用动态掩码而不是静态掩码

#### 使用示例

```python
from transformers import RobertaTokenizer, RobertaModel
import torch

# 加载预训练的RoBERTa模型和分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# 准备输入
text = "RoBERTa is an optimized version of BERT."
encoded_input = tokenizer(text, return_tensors='pt')

# 获取模型输出
with torch.no_grad():
    output = model(**encoded_input)

# 获取最后隐藏状态
sequence_output = output.last_hidden_state
print(f"Sequence output shape: {sequence_output.shape}")
```

### T5 (Text-to-Text Transfer Transformer)

T5是一个编码器-解码器模型，将所有NLP任务统一为文本到文本的转换任务。

#### 特点
- 统一的文本到文本框架
- 可以处理多种NLP任务
- 通过任务前缀指定要执行的任务

#### 使用示例

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载预训练的T5模型和分词器
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# 翻译任务
input_text = "translate English to German: Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

# 生成翻译
outputs = model.generate(input_ids, max_length=40)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translated text: {translated_text}")

# 摘要任务
input_text = "summarize: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Locally nicknamed 'La dame de fer', it was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair."
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

# 生成摘要
outputs = model.generate(input_ids, max_length=40, min_length=10)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Summary: {summary}")
```

### DistilBERT

DistilBERT是BERT的轻量级版本，通过知识蒸馏技术减小了模型大小，同时保持了相当的性能。

#### 特点
- 比BERT小40%
- 速度提高60%
- 保留了BERT 97%的性能

#### 使用示例

```python
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# 加载预训练的DistilBERT模型和分词器
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# 准备输入
text = "DistilBERT is a smaller and faster version of BERT."
encoded_input = tokenizer(text, return_tensors='pt')

# 获取模型输出
with torch.no_grad():
    output = model(**encoded_input)

# 获取最后隐藏状态
sequence_output = output.last_hidden_state
print(f"Sequence output shape: {sequence_output.shape}")
```

### BART (Bidirectional and Auto-Regressive Transformers)

BART是一个编码器-解码器模型，特别适合文本生成、翻译和摘要任务。

#### 特点
- 结合了BERT的双向编码器和GPT的自回归解码器
- 适合序列到序列任务
- 对噪声文本有很强的鲁棒性

#### 使用示例

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# 加载预训练的BART模型和分词器
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# 准备输入文本（用于摘要）
text = """
The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, 
and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. 
During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest 
man-made structure in the world, a title it held for 41 years until the Chrysler Building in 
New York City was finished in 1930.
"""

# 编码输入文本
inputs = tokenizer([text], max_length=1024, return_tensors='pt')

# 生成摘要
summary_ids = model.generate(
    inputs.input_ids, 
    num_beams=4,
    max_length=100,
    min_length=30,
    early_stopping=True
)

# 解码摘要
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"Summary: {summary}")
```

### ViT (Vision Transformer)

ViT是一个用于图像分类的Transformer模型，它将图像分割成小块并像处理序列一样处理它们。

#### 特点
- 将图像视为一系列的图像块序列
- 不使用卷积层
- 在大规模数据集上训练时表现优异

#### 使用示例

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

# 加载预训练的ViT模型和特征提取器
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 加载图像
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'  # 一张猫的图片
image = Image.open(requests.get(url, stream=True).raw)

# 准备输入
inputs = feature_extractor(images=image, return_tensors="pt")

# 获取预测
import torch
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测的类别
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
```

## 模型配置和自定义

### 修改模型配置

你可以在加载模型时修改其配置：

```python
from transformers import BertConfig, BertModel

# 创建自定义配置
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=6,  # 减少层数
    num_attention_heads=12,
    intermediate_size=3072
)

# 使用自定义配置创建模型
model = BertModel(config)
```

### 从头训练模型

你也可以从头开始训练模型，而不是使用预训练权重：

```python
from transformers import BertConfig, BertForMaskedLM

# 创建配置
config = BertConfig()

# 创建未初始化的模型
model = BertForMaskedLM(config)
```

## 模型保存和加载

### 保存模型

```python
# 保存模型和分词器
model.save_pretrained('./my_model_directory')
tokenizer.save_pretrained('./my_model_directory')
```

### 加载保存的模型

```python
from transformers import AutoModel, AutoTokenizer

# 加载保存的模型和分词器
model = AutoModel.from_pretrained('./my_model_directory')
tokenizer = AutoTokenizer.from_pretrained('./my_model_directory')
```

## 模型推理优化

### 使用半精度浮点数(FP16)

```python
import torch
from transformers import AutoModelForSequenceClassification

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

# 转换为半精度
model = model.half()  # 将模型转换为FP16
```

### 使用ONNX Runtime

```python
from transformers import AutoTokenizer
from transformers.convert_graph_to_onnx import convert_graph_to_onnx
import onnxruntime as ort
import numpy as np

# 转换模型到ONNX格式
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
convert_graph_to_onnx(
    framework="pt",
    model=model_name,
    output="model.onnx",
    opset=11
)

# 加载分词器和ONNX模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
ort_session = ort.InferenceSession("model.onnx")

# 准备输入
text = "I love this movie!"
inputs = tokenizer(text, return_tensors="np")

# 运行推理
ort_inputs = {k: v for k, v in inputs.items()}
ort_outputs = ort_session.run(None, ort_inputs)

# 处理输出
logits = ort_outputs[0]
predictions = np.argmax(logits, axis=1)
print(f"Prediction: {predictions}")
```

### 使用TorchScript

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载模型和分词器
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备示例输入
text = "I love this movie!"
inputs = tokenizer(text, return_tensors="pt")

# 转换为TorchScript
traced_model = torch.jit.trace(model, [inputs.input_ids, inputs.attention_mask])

# 保存TorchScript模型
traced_model.save("model_torchscript.pt")

# 加载TorchScript模型
loaded_model = torch.jit.load("model_torchscript.pt")

# 使用加载的模型进行推理
with torch.no_grad():
    outputs = loaded_model(inputs.input_ids, inputs.attention_mask)
    
predictions = torch.nn.functional.softmax(outputs[0], dim=-1)
print(f"Predictions: {predictions}")
```

## 模型量化

模型量化可以显著减小模型大小并加速推理，但可能会略微降低准确性。

```python
from transformers import AutoModelForSequenceClassification
import torch

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# 动态量化（PyTorch 1.3+）
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 使用量化模型进行推理
# ...
```

## 多GPU和分布式训练

### 使用DataParallel

```python
import torch
from transformers import AutoModelForSequenceClassification

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 使用DataParallel包装模型
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model = model.to('cuda')  # 将模型移动到GPU
```

### 使用DistributedDataParallel

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForSequenceClassification

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = model.to(torch.device('cuda'))

# 使用DistributedDataParallel包装模型
model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
```

## 模型部署

### 使用TorchServe

TorchServe是PyTorch的模型服务框架，可以用来部署Transformers模型。

```bash
# 安装TorchServe
pip install torchserve torch-model-archiver torch-workflow-archiver

# 创建模型存档
torch-model-archiver --model-name bert_model \
                     --version 1.0 \
                     --serialized-file path/to/model.pt \
                     --handler transformers_handler.py \
                     --extra-files "path/to/tokenizer/,path/to/config.json"

# 启动TorchServe
torchserve --start --model-store model_store --models bert=bert_model.mar
```

### 使用FastAPI

FastAPI是一个现代、快速的Web框架，适合创建API。

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# 加载模型和分词器
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: TextInput):
    # 准备输入
    inputs = tokenizer(input_data.text, return_tensors="pt")
    
    # 进行推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 处理输出
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_id = predictions.argmax().item()
    label = model.config.id2label[label_id]
    score = predictions[0][label_id].item()
    
    return {"label": label, "score": score}
```

## 常见问题与解决方案

### 1. 内存不足

当处理大型模型或长序列时，可能会遇到内存不足的问题。

解决方案：
- 使用较小的批量大小
- 使用梯度累积
- 使用混合精度训练
- 使用较小的模型（如DistilBERT代替BERT）
- 使用模型并行或分布式训练

### 2. 推理速度慢

解决方案：
- 使用半精度浮点数(FP16)
- 使用模型量化
- 使用ONNX Runtime或TorchScript
- 使用较小的模型
- 使用批处理而不是单个样本处理

### 3. 模型过拟合

解决方案：
- 使用更多的训练数据
- 使用正则化技术（如权重衰减、Dropout）
- 使用早停法
- 减小模型大小

## 下一步

现在你已经了解了如何使用各种预训练模型，可以继续学习：

1. 查看 `pipeline_usage.md` 了解如何使用pipeline API简化工作流程
2. 学习 `fine_tuning.md` 了解如何微调模型以适应特定任务
3. 探索 `examples/` 目录中的实际应用示例