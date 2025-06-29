# Transformers 高级主题

本文档涵盖了 Hugging Face Transformers 库的高级使用主题，包括自定义模型架构、分布式训练、模型压缩等技术。

## 1. 自定义模型架构

### 1.1 创建自定义配置

```python
from transformers import PretrainedConfig

class MyCustomConfig(PretrainedConfig):
    model_type = "my-custom-model"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
```

### 1.2 创建自定义模型

```python
from transformers import PreTrainedModel
import torch.nn as nn

class MyCustomModel(PreTrainedModel):
    config_class = MyCustomConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout_prob
            ),
            num_layers=config.num_hidden_layers
        )
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embeddings)
        return encoder_outputs
```

## 2. 分布式训练

### 2.1 使用 PyTorch DDP

```python
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
import torch

def setup_ddp(rank, world_size):
    init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )

def train_ddp(rank, world_size, model, train_dataset):
    setup_ddp(rank, world_size)
    
    # 将模型移动到当前设备
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    
    # 包装模型用于DDP
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # 创建数据加载器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            # 训练步骤
            pass
```

### 2.2 使用 Accelerate

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def train_with_accelerate():
    accelerator = Accelerator()
    
    # 加载模型和数据
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    train_dataloader = get_train_dataloader()
    
    # 准备训练
    model, train_dataloader = accelerator.prepare(model, train_dataloader)
    
    # 训练循环
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
```

## 3. 模型压缩技术

### 3.1 知识蒸馏

```python
from transformers import AutoModelForSequenceClassification

class DistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=2.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
    
    def compute_distillation_loss(self, teacher_logits, student_logits, labels):
        # 硬损失
        hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
        
        # 软损失
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = nn.functional.softmax(student_logits / self.temperature, dim=-1)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            nn.functional.log_softmax(student_logits / self.temperature, dim=-1),
            soft_targets
        ) * (self.temperature ** 2)
        
        return hard_loss + soft_loss
```

### 3.2 量化

```python
from transformers import AutoModelForSequenceClassification
import torch.quantization

def quantize_model():
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    
    # 准备量化
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # 校准（使用一些校准数据）
    calibrate_model(model)
    
    # 转换为量化模型
    torch.quantization.convert(model, inplace=True)
    
    return model
```

### 3.3 剪枝

```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    # 对所有线性层应用L1范数剪枝
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    
    # 使剪枝永久化
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    
    return model
```

## 4. 高级优化技术

### 4.1 梯度累积和梯度裁剪

```python
def train_with_gradient_accumulation(model, train_dataloader, accumulation_steps=4):
    model.zero_grad()
    for i, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.zero_grad()
```

### 4.2 自定义学习率调度器

```python
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
```

## 5. 模型部署最佳实践

### 5.1 模型导出为ONNX

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def export_to_onnx(model_name, onnx_path):
    # 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 准备示例输入
    text = "Example text for ONNX export"
    inputs = tokenizer(text, return_tensors="pt")
    
    # 导出模型
    torch.onnx.export(
        model,
        (inputs.input_ids, inputs.attention_mask),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        },
        opset_version=12
    )
```

### 5.2 TorchScript转换

```python
def convert_to_torchscript(model):
    # 准备示例输入
    example_input_ids = torch.randint(1000, (1, 128))
    example_attention_mask = torch.ones((1, 128))
    
    # 转换为TorchScript
    traced_model = torch.jit.trace(
        model,
        (example_input_ids, example_attention_mask)
    )
    
    return traced_model
```

## 6. 调试和性能优化

### 6.1 内存优化

```python
def optimize_memory_usage(model, training_args):
    # 使用梯度检查点
    model.gradient_checkpointing_enable()
    
    # 使用混合精度训练
    training_args.fp16 = True
    
    # 使用梯度累积
    training_args.gradient_accumulation_steps = 4
    
    return model, training_args
```

### 6.2 性能分析

```python
import torch.profiler

def profile_model_performance(model, sample_input):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # 运行模型
        outputs = model(**sample_input)
    
    # 打印分析结果
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=10))
```

## 7. 下一步学习建议

1. 探索更多的预训练模型架构和任务
2. 学习如何创建和训练自定义模型架构
3. 实践分布式训练和模型并行化
4. 研究模型压缩和优化技术
5. 了解生产环境中的部署策略

## 8. 参考资源

- [Hugging Face 文档](https://huggingface.co/docs)
- [PyTorch 分布式训练指南](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Transformers 性能优化指南](https://huggingface.co/docs/transformers/performance)
- [模型压缩最佳实践](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer)
