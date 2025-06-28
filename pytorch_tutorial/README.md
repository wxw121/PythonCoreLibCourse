# PyTorch 学习指南

这个目录包含了全面的PyTorch学习资料，从入门到进阶，帮助你掌握PyTorch和深度学习。

## 文档目录

### [1. 快速入门指南](quick_start.md)
适合初学者的快速上手教程，包含：
- 环境设置
- 基础操作示例
- 常见任务示例
- 调试技巧
- 常见问题解决
- 性能优化
- 学习建议

### [2. 综合指南](comprehensive_guide.md)
深入的PyTorch和深度学习教程，包含：
- PyTorch核心API详解
- 深度学习基础概念
- 常见模型架构
- 实践建议
- 实际应用场景
- 性能优化指南
- 调试与监控
- 最佳实践总结

## 代码示例

### 1. [PyTorch基础](pytorch_basics.py)
- 张量操作
- 自动求导
- 基本神经网络组件

### 2. [神经网络示例](pytorch_neural_networks.py)
- 完整的神经网络实现
- 训练循环
- 模型评估

### 3. [计算机视觉示例](pytorch_computer_vision.py)
- 图像处理
- CNN模型
- 迁移学习

### 4. [自然语言处理示例](pytorch_nlp.py)
- 文本处理
- RNN/LSTM模型
- 词嵌入

## 学习路径建议

### 1. 新手入门（1-2周）
1. 阅读 [快速入门指南](quick_start.md)
2. 运行并理解 [PyTorch基础](pytorch_basics.py)
3. 完成基础示例的练习

### 2. 基础夯实（2-4周）
1. 学习 [综合指南](comprehensive_guide.md) 的基础概念部分
2. 实现 [神经网络示例](pytorch_neural_networks.py)
3. 尝试修改和优化模型

### 3. 应用实践（4-8周）
1. 深入学习 [综合指南](comprehensive_guide.md) 的高级主题
2. 研究 [计算机视觉示例](pytorch_computer_vision.py) 和 [自然语言处理示例](pytorch_nlp.py)
3. 开始自己的项目实践

### 4. 进阶提高（持续学习）
1. 阅读论文实现经典模型
2. 参与开源项目
3. 在实际项目中应用所学知识

## 常见问题

### 1. 运行环境
- Python 3.6+
- PyTorch 1.8+
- CUDA（可选，用于GPU加速）

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. GPU支持
确保你的CUDA版本与PyTorch版本兼容。可以通过以下代码检查：
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## 使用方法

### 交互式模式

运行主入口文件，选择你想要学习的模块：

```bash
python pytorch_tutorial.py
```

这将显示一个交互式菜单，你可以选择要运行的教程模块。

### 命令行模式

你也可以直接指定要运行的模块：

```bash
# 运行PyTorch基础教程
python pytorch_tutorial.py 1

# 运行PyTorch神经网络教程
python pytorch_tutorial.py 2

# 运行PyTorch计算机视觉教程
python pytorch_tutorial.py 3

# 运行PyTorch自然语言处理教程
python pytorch_tutorial.py 4

# 运行所有教程
python pytorch_tutorial.py 5
```

### 单独运行模块

你也可以直接运行各个模块：

```bash
python pytorch_basics.py
python pytorch_neural_networks.py
python pytorch_computer_vision.py
python pytorch_nlp.py
```

## 模块说明

### 1. PyTorch基础 (pytorch_basics.py)

- 张量创建与操作
- 自动求导机制
- GPU加速计算
- 数据加载与预处理

### 2. PyTorch神经网络 (pytorch_neural_networks.py)

- 线性层和激活函数
- 构建简单神经网络
- 损失函数与优化器
- 训练和验证流程

### 3. PyTorch计算机视觉 (pytorch_computer_vision.py)

- 图像数据处理
- 卷积神经网络基础
- 图像分类实战
- 数据增强技术
- 迁移学习

### 4. PyTorch自然语言处理 (pytorch_nlp.py)

- 文本数据处理
- 词嵌入(Word Embeddings)
- 循环神经网络(RNN)
- LSTM和GRU
- Transformer基础

## 学习路径

如果你是PyTorch新手，建议按照以下顺序学习：

1. 首先学习PyTorch基础，了解张量和自动求导
2. 然后学习PyTorch神经网络，掌握构建和训练神经网络的基本流程
3. 根据你的兴趣，选择计算机视觉或自然语言处理进行深入学习

## 注意事项

- 教程中的示例代码主要用于演示概念，可能需要根据实际应用进行调整
- 为了提高学习效果，建议在学习过程中尝试修改代码参数，观察结果变化
- 如果你有GPU，可以通过设置`device = torch.device("cuda")`来加速模型训练


## 贡献指南

欢迎提交问题和改进建议！请：
1. Fork本仓库
2. 创建你的特性分支
3. 提交你的改动
4. 创建Pull Request

## 资源链接

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch论坛](https://discuss.pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [深度学习论文](https://paperswithcode.com/)

## 许可证

本教程采用MIT许可证。详见[LICENSE](LICENSE)文件。


