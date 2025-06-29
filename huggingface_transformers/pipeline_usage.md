# Hugging Face Transformers Pipeline API 使用指南

## 什么是Pipeline API？

Pipeline API是Hugging Face Transformers库提供的一个高级接口，它将预处理、模型推理和后处理步骤封装在一起，使得执行常见任务变得非常简单。使用Pipeline API，你可以用几行代码就完成复杂的NLP或计算机视觉任务，而无需关心底层的细节。

## Pipeline API的优势

- **简单易用**：只需几行代码即可完成复杂任务
- **灵活性**：支持多种任务和模型
- **可定制**：可以指定使用的模型、分词器和参数
- **一致的接口**：不同任务使用相似的API，易于学习和使用

## 支持的任务

Pipeline API支持多种自然语言处理和计算机视觉任务，包括但不限于：

1. 文本分类（如情感分析）
2. 命名实体识别
3. 问答
4. 文本生成
5. 文本摘要
6. 机器翻译
7. 特征提取
8. 填充掩码
9. 图像分类
10. 目标检测
11. 图像分割
12. 零样本分类

## 基本用法

### 创建Pipeline

创建pipeline的基本语法如下：

```python
from transformers import pipeline

# 创建一个pipeline
pipe = pipeline(task, model=model_name, **kwargs)
```

其中：
- `task`：要执行的任务名称
- `model_name`：要使用的预训练模型名称（可选）
- `**kwargs`：其他参数，如设备、批量大小等

### 使用Pipeline

创建pipeline后，可以直接将输入数据传递给它：

```python
# 使用pipeline
result = pipe(inputs)
```

## 常见任务示例

### 1. 情感分析

情感分析是判断文本情感倾向（如正面、负面或中性）的任务。

```python
from transformers import pipeline

# 创建情感分析pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# 分析单个文本
text = "I really enjoyed this movie. The plot was engaging and the acting was superb!"
result = sentiment_analyzer(text)
print(result)
# 输出: [{'label': 'POSITIVE', 'score': 0.9998}]

# 分析多个文本
texts = [
    "I really enjoyed this movie.",
    "This was a terrible waste of time.",
    "The movie was okay, nothing special."
]
results = sentiment_analyzer(texts)
for i, result in enumerate(results):
    print(f"Text {i+1}: {result['label']} (Score: {result['score']:.4f})")
```

#### 使用特定模型

```python
# 使用特定的情感分析模型
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
```

#### 处理长文本

```python
# 处理长文本
long_text = "..." # 一段很长的文本
# 设置truncation=True来截断过长的文本
result = sentiment_analyzer(long_text, truncation=True)
```

### 2. 命名实体识别

命名实体识别(NER)是识别文本中的实体（如人名、地名、组织名等）的任务。

```python
from transformers import pipeline

# 创建命名实体识别pipeline
ner = pipeline("ner")

# 识别文本中的实体
text = "Hugging Face was founded in Paris, France by Clément Delangue and Julien Chaumond in 2016."
results = ner(text)

# 处理结果
entities = {}
for result in results:
    entity_group = result['entity_group']
    word = result['word']
    score = result['score']
    
    if entity_group not in entities:
        entities[entity_group] = []
    
    entities[entity_group].append((word, score))

# 打印识别出的实体
for entity_type, words in entities.items():
    print(f"{entity_type}: {', '.join([word for word, _ in words])}")
```

#### 分组实体

```python
from transformers import pipeline
import numpy as np

# 创建命名实体识别pipeline
ner = pipeline("ner", aggregation_strategy="simple")

# 识别文本中的实体
text = "Hugging Face was founded in Paris, France by Clément Delangue and Julien Chaumond in 2016."
results = ner(text)

# 打印分组后的实体
for entity in results:
    print(f"{entity['entity_group']}: {entity['word']} (Score: {entity['score']:.4f})")
```

### 3. 问答

问答任务是根据给定的上下文回答问题。

```python
from transformers import pipeline

# 创建问答pipeline
qa = pipeline("question-answering")

# 准备上下文和问题
context = """
Hugging Face is an AI community and machine learning platform that provides tools 
that enable users to build, train and deploy machine learning models based on open 
source code and technologies. The company was founded in 2016 by Clément Delangue, 
Julien Chaumond, and Thomas Wolf. Hugging Face was originally founded to develop 
a chatbot app targeted at teenagers, which used natural language processing to 
understand the messages it received and generate appropriate responses.
"""

questions = [
    "When was Hugging Face founded?",
    "Who founded Hugging Face?",
    "What does Hugging Face provide?"
]

# 回答问题
for question in questions:
    result = qa(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']} (Score: {result['score']:.4f})")
    print(f"Start: {result['start']}, End: {result['end']}")
    print()
```

#### 使用特定模型

```python
# 使用特定的问答模型
qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)
```

### 4. 文本生成

文本生成任务是根据给定的提示生成连贯的文本。

```python
from transformers import pipeline

# 创建文本生成pipeline
generator = pipeline("text-generation")

# 生成文本
prompt = "Once upon a time in a land far, far away,"
result = generator(
    prompt,
    max_length=50,
    num_return_sequences=3,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

# 打印生成的文本
for i, sequence in enumerate(result):
    print(f"Sequence {i+1}: {sequence['generated_text']}")
```

#### 控制生成参数

```python
# 控制生成参数
result = generator(
    "Artificial intelligence is",
    max_length=100,        # 生成文本的最大长度
    min_length=30,         # 生成文本的最小长度
    do_sample=True,        # 使用采样而不是贪婪解码
    temperature=0.9,       # 控制随机性（较高值 = 更随机）
    top_k=50,              # 只考虑概率最高的前k个词
    top_p=0.95,            # 核采样（只考虑累积概率达到p的词）
    repetition_penalty=1.2 # 惩罚重复的词
)
```

### 5. 文本摘要

文本摘要任务是将长文本压缩成更短的版本，同时保留关键信息。

```python
from transformers import pipeline

# 创建文本摘要pipeline
summarizer = pipeline("summarization")

# 准备长文本
article = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower. 
Locally nicknamed "La dame de fer" (French for "Iron Lady"), it was constructed from 1887 to 1889 
as the entrance to the 1889 World's Fair. It was initially criticised by some of France's leading 
artists and intellectuals for its design, but it has become a global cultural icon of France and 
one of the most recognisable structures in the world. The Eiffel Tower is the most-visited paid 
monument in the world; 6.91 million people ascended it in 2015. The tower is 330 metres (1,083 ft) 
tall, about the same height as an 81-storey building, and the tallest structure in Paris. 
Its base is square, measuring 125 metres (410 ft) on each side.
"""

# 生成摘要
summary = summarizer(
    article,
    max_length=150,
    min_length=30,
    do_sample=False
)

print(f"Summary: {summary[0]['summary_text']}")
```

#### 使用特定模型

```python
# 使用特定的摘要模型
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)
```

### 6. 机器翻译

机器翻译任务是将文本从一种语言翻译成另一种语言。

```python
from transformers import pipeline

# 创建翻译pipeline
translator = pipeline("translation_en_to_fr")

# 翻译文本
text = "Hugging Face is a company that provides NLP tools and models."
result = translator(text)

print(f"Original: {text}")
print(f"Translation: {result[0]['translation_text']}")
```

#### 使用特定模型和语言对

```python
# 使用特定的翻译模型（英语到德语）
translator_en_de = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-de"
)

# 翻译文本
text = "Hugging Face is a company that provides NLP tools and models."
result = translator_en_de(text)
print(f"English to German: {result[0]['translation_text']}")

# 使用特定的翻译模型（英语到中文）
translator_en_zh = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-zh"
)

# 翻译文本
result = translator_en_zh(text)
print(f"English to Chinese: {result[0]['translation_text']}")
```

### 7. 特征提取

特征提取任务是从文本中提取向量表示，这些向量可以用于下游任务如文本相似度计算、聚类等。

```python
from transformers import pipeline
import numpy as np

# 创建特征提取pipeline
feature_extractor = pipeline("feature-extraction")

# 准备文本
texts = [
    "Hugging Face is a company that provides NLP tools.",
    "The company was founded in 2016.",
    "Transformers is their most popular library."
]

# 提取特征
features = feature_extractor(texts)

# 查看特征形状
for i, text_features in enumerate(features):
    print(f"Text {i+1} features shape: {np.array(text_features).shape}")

# 计算文本相似度（使用第一个token的表示）
from sklearn.metrics.pairwise import cosine_similarity

# 提取每个文本的[CLS]标记表示（第一个token）
text_vectors = [np.array(text_features)[0][0] for text_features in features]

# 计算余弦相似度
similarity_matrix = cosine_similarity(text_vectors)
print("\nSimilarity Matrix:")
for i in range(len(texts)):
    for j in range(len(texts)):
        print(f"Similarity between text {i+1} and text {j+1}: {similarity_matrix[i][j]:.4f}")
```

### 8. 填充掩码

填充掩码任务是预测文本中被掩盖的词。

```python
from transformers import pipeline

# 创建填充掩码pipeline
unmasker = pipeline("fill-mask")

# 准备带掩码的文本
text = "Hugging Face is a [MASK] that provides NLP tools and models."

# 填充掩码
results = unmasker(text)

# 打印结果
print(f"Original text: {text}")
print("Top predictions:")
for result in results:
    print(f"- {result['token_str']} (Score: {result['score']:.4f})")
```

#### 使用不同的掩码标记

不同的模型使用不同的掩码标记：
- BERT: `[MASK]`
- RoBERTa: `<mask>`
- DistilBERT: `[MASK]`
- ALBERT: `[MASK]`

```python
# 使用RoBERTa模型
unmasker = pipeline("fill-mask", model="roberta-base")
text = "Hugging Face is a <mask> that provides NLP tools and models."
results = unmasker(text)
```

### 9. 图像分类

图像分类任务是识别图像中的主要对象或场景。

```python
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# 创建图像分类pipeline
image_classifier = pipeline("image-classification")

# 从URL加载图像
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 一张猫的图片
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# 分类图像
results = image_classifier(image)

# 打印结果
print("Top predictions:")
for result in results:
    print(f"- {result['label']} (Score: {result['score']:.4f})")
```

#### 使用本地图像

```python
# 加载本地图像
local_image = Image.open("path/to/your/image.jpg")
results = image_classifier(local_image)
```

### 10. 零样本分类

零样本分类允许你在没有特定训练数据的情况下，使用自然语言标签对文本进行分类。

```python
from transformers import pipeline

# 创建零样本分类pipeline
classifier = pipeline("zero-shot-classification")

# 准备文本和候选标签
text = "This restaurant serves delicious food at reasonable prices in a cozy atmosphere."
candidate_labels = ["price", "quality", "ambiance", "service"]

# 进行分类
result = classifier(text, candidate_labels)

# 打印结果
print(f"Text: {text}")
print("\nClassification results:")
for label, score in zip(result['labels'], result['scores']):
    print(f"- {label}: {score:.4f}")
```

#### 多标签分类

```python
# 多标签分类
result = classifier(
    text,
    candidate_labels,
    multi_label=True  # 允许多个标签同时适用
)

print("\nMulti-label classification results:")
for label, score in zip(result['labels'], result['scores']):
    print(f"- {label}: {score:.4f}")
```

### 11. 目标检测

目标检测任务是识别图像中的对象及其位置。

```python
from transformers import pipeline
from PIL import Image, ImageDraw
import requests
from io import BytesIO

# 创建目标检测pipeline
object_detector = pipeline("object-detection")

# 从URL加载图像
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# 检测对象
results = object_detector(image)

# 打印结果
print("Detected objects:")
for result in results:
    print(f"- {result['label']} (Score: {result['score']:.4f}) at {result['box']}")

# 在图像上绘制边界框
draw = ImageDraw.Draw(image)
for result in results:
    box = result['box']
    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="red", width=3)
    draw.text((box['xmin'], box['ymin']), f"{result['label']}: {result['score']:.2f}", fill="white")

# 显示图像
image.show()
```

### 12. 音频分类

音频分类任务是识别音频中的声音类型或内容。

```python
from transformers import pipeline
import librosa

# 创建音频分类pipeline
audio_classifier = pipeline("audio-classification")

# 加载音频文件
audio_file = "path/to/audio.wav"  # 替换为你的音频文件路径
audio, _ = librosa.load(audio_file, sr=16000)

# 分类音频
results = audio_classifier(audio)

# 打印结果
print("Audio classification results:")
for result in results:
    print(f"- {result['label']}: {result['score']:.4f}")
```

## 高级用法

### 1. 批处理

处理大量数据时，使用批处理可以提高效率：

```python
from transformers import pipeline
import torch

# 创建pipeline时指定批量大小
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    device=0 if torch.cuda.is_available() else -1,  # 使用GPU（如果可用）
    batch_size=32  # 设置批量大小
)

# 准备大量文本
texts = ["Text " + str(i) for i in range(100)]  # 示例文本

# 批量处理
results = sentiment_analyzer(texts)
```

### 2. 指定设备

可以指定pipeline使用的设备（CPU或GPU）：

```python
from transformers import pipeline
import torch

# 使用GPU（如果可用）
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", device=device)

# 或者明确指定GPU设备
# classifier = pipeline("sentiment-analysis", device=0)  # 使用第一个GPU
```

### 3. 自定义模型和分词器

可以为pipeline指定自定义的模型和分词器：

```python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# 加载自定义模型和分词器
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建使用自定义模型和分词器的pipeline
classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)
```

### 4. 处理长文本

处理超过模型最大上下文长度的文本：

```python
from transformers import pipeline

# 创建pipeline
summarizer = pipeline("summarization")

# 处理长文本的函数
def summarize_long_text(text, max_chunk_length=1000, overlap=100):
    # 将文本分成重叠的块
    chunks = []
    for i in range(0, len(text), max_chunk_length - overlap):
        chunk = text[i:i + max_chunk_length]
        chunks.append(chunk)
    
    # 对每个块进行摘要
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30)[0]['summary_text']
        summaries.append(summary)
    
    # 将所有摘要合并
    if len(summaries) == 1:
        return summaries[0]
    else:
        # 递归地对摘要进行摘要，直到得到一个最终摘要
        combined_summary = " ".join(summaries)
        if len(combined_summary) > max_chunk_length:
            return summarize_long_text(combined_summary, max_chunk_length, overlap)
        else:
            return summarizer(combined_summary, max_length=150, min_length=30)[0]['summary_text']

# 使用函数处理长文本
long_text = "..." # 一段很长的文本
final_summary = summarize_long_text(long_text)
print(f"Final summary: {final_summary}")
```

## 创建自定义Pipeline

你可以创建自定义pipeline来处理特定任务：

```python
from transformers import Pipeline, AutoModelForSequenceClassification, AutoTokenizer

class CustomPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        # 处理参数
        preprocess_kwargs = {}
        if "example_param" in kwargs:
            preprocess_kwargs["example_param"] = kwargs["example_param"]
        
        postprocess_kwargs = {}
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        
        return preprocess_kwargs, {}, postprocess_kwargs
    
    def preprocess(self, inputs, example_param=None):
        # 预处理输入
        return {"inputs": self.tokenizer(inputs, return_tensors="pt")}
    
    def _forward(self, model_inputs):
        # 模型前向传播
        return self.model(**model_inputs)
    
    def postprocess(self, model_outputs, threshold=0.5):
        # 后处理输出
        logits = model_outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # 应用阈值
        predictions = []
        for prob in probabilities:
            label_idx = prob.argmax().item()
            score = prob[label_idx].item()
            if score >= threshold:
                predictions.append({
                    "label": self.model.config.id2label[label_idx],
                    "score": score
                })
            else:
                predictions.append({
                    "label": "UNCERTAIN",
                    "score": score
                })
        
        return predictions

# 使用自定义pipeline
from transformers import pipeline

# 加载模型和分词器
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建自定义pipeline
custom_pipe = pipeline(
    task="custom-classification",
    model=model,
    tokenizer=tokenizer,
    pipeline_class=CustomPipeline
)

# 使用自定义pipeline
result = custom_pipe("I love this movie!", threshold=0.8)
print(result)
```

## 常见问题与解决方案

### 1. 内存不足

当处理大型模型或大量数据时，可能会遇到内存不足的问题。

解决方案：
- 减小批量大小
- 使用较小的模型
- 使用CPU而不是GPU（如果GPU内存有限）
- 使用模型量化

### 2. 处理速度慢

解决方案：
- 增加批量大小（如果内存允许）
- 使用GPU加速
- 使用较小的模型
- 使用ONNX Runtime或TorchScript优化

### 3. 处理特殊字符或非英语文本

解决方案：
- 使用多语言模型（如`xlm-roberta-base`）
- 确保正确处理文本编码
- 对于特定语言，使用针对该语言训练的模型

## 下一步

现在你已经了解了如何使用Pipeline API，可以继续学习：

1. 查看 `fine_tuning.md` 了解如何微调模型以适应特定任务
2. 探索 `examples/` 目录中的实际应用示例
3. 学习 `advanced_topics.md` 了解更高级的技巧和最佳实践