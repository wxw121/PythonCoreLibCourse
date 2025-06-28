# PyTorch 快速入门指南

## 1. 环境设置
```bash
# 安装PyTorch
pip install torch torchvision torchaudio

# 验证安装
python -c "import torch; print(torch.__version__)"
```

## 2. 基础操作示例

### 2.1 创建张量
```python
import torch

# 从列表创建张量
x = torch.tensor([1, 2, 3])

# 创建随机张量
random_tensor = torch.rand(3, 3)

# 创建特定值张量
zeros = torch.zeros(2, 2)
ones = torch.ones(2, 2)
```

### 2.2 简单神经网络
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 创建模型实例
model = SimpleNet()
```

### 2.3 训练循环示例
```python
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

## 3. 常见任务示例

### 3.1 图像分类
```python
import torchvision
import torchvision.transforms as transforms

# 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=4,
    shuffle=True
)
```

### 3.2 文本分类
```python
from torch.nn.utils.rnn import pack_padded_sequence

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)  # 二分类
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])
```

## 4. 调试技巧

### 4.1 张量形状检查
```python
def shape_check(x, name="tensor"):
    print(f"{name}.shape:", x.shape)
    print(f"{name}.dtype:", x.dtype)
    print(f"{name}.device:", x.device)
```

### 4.2 梯度检查
```python
def grad_check(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad stats:")
            print(f"- mean: {param.grad.mean()}")
            print(f"- std: {param.grad.std()}")
```

## 5. 常见问题解决

### 5.1 CUDA内存问题
```python
# 清理GPU缓存
torch.cuda.empty_cache()

# 使用梯度累积减少内存使用
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5.2 模型保存和加载
```python
# 保存模型
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# 加载模型
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 6. 性能优化

### 6.1 使用GPU加速
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
inputs = inputs.to(device)
```

### 6.2 数据加载优化
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # 多进程加载
    pin_memory=True  # GPU训练时使用
)
```

## 7. 下一步学习建议

1. 深入学习PyTorch文档
2. 实现经典论文
3. 参与开源项目
4. 在Kaggle上实践
5. 阅读[comprehensive_guide.md](comprehensive_guide.md)获取更多详细信息

记住：实践是最好的学习方式。从简单项目开始，逐步增加复杂度。遇到问题时，查看文档，使用调试工具，向社区寻求帮助。
