#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch神经网络基础教程

本模块涵盖PyTorch中神经网络的基础概念和实现，包括：
1. 线性层和激活函数
2. 构建简单神经网络
3. 损失函数和优化器
4. 训练和验证流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scikit_learn_tutorial.config import set_matplotlib_chinese
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Dict, Any, Optional


def linear_layer_basics() -> None:
    """
    线性层和激活函数基础

    展示PyTorch中线性层的使用方法和常见激活函数
    """
    print("\n" + "=" * 50)
    print("线性层和激活函数基础".center(50))
    print("=" * 50)

    # 1. 线性层基础
    print("\n1. 线性层基础:")

    # 创建一个简单的线性层
    in_features = 10
    out_features = 5
    linear = nn.Linear(in_features, out_features)

    print(f"线性层结构:\n{linear}")
    print(f"权重形状: {linear.weight.shape}")
    print(f"偏置形状: {linear.bias.shape}")

    # 使用线性层
    x = torch.randn(3, in_features)  # 批量大小为3
    y = linear(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # 2. 常见激活函数
    print("\n2. 常见激活函数:")

    # 准备数据
    x = torch.linspace(-5, 5, 100)

    # 创建激活函数
    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    leaky_relu = nn.LeakyReLU(0.1)

    # 应用激活函数
    y_relu = relu(x)
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)
    y_leaky_relu = leaky_relu(x)

    # 绘制激活函数
    plt.figure(figsize=(12, 6))
    plt.plot(x.numpy(), y_relu.numpy(), label='ReLU')
    plt.plot(x.numpy(), y_sigmoid.numpy(), label='Sigmoid')
    plt.plot(x.numpy(), y_tanh.numpy(), label='Tanh')
    plt.plot(x.numpy(), y_leaky_relu.numpy(), label='Leaky ReLU')
    plt.grid(True)
    plt.legend()
    plt.title('常见激活函数')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.close()

    print("常见激活函数及其特点:")
    print("- ReLU: 最常用的激活函数，对负值输出0")
    print("- Sigmoid: 将输入压缩到(0,1)区间，常用于二分类")
    print("- Tanh: 将输入压缩到(-1,1)区间")
    print("- Leaky ReLU: ReLU的变体，对负值有小斜率")


def simple_neural_network() -> None:
    """
    构建简单神经网络

    展示如何使用PyTorch构建、训练和评估简单的神经网络
    """
    print("\n" + "=" * 50)
    print("构建简单神经网络".center(50))
    print("=" * 50)

    # 1. 定义神经网络类
    class SimpleNN(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            """
            初始化神经网络

            Args:
                input_size: 输入特征维度
                hidden_size: 隐藏层神经元数量
                output_size: 输出维度
            """
            super().__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            前向传播

            Args:
                x: 输入数据

            Returns:
                模型输出
            """
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    # 2. 创建模型实例
    print("\n1. 创建模型:")
    model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
    print(f"模型结构:\n{model}")

    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"总参数量: {total_params}")
    print(f"可训练参数量: {trainable_params}")

    # 3. 创建示例数据
    print("\n2. 准备数据:")

    # 创建随机数据
    X = torch.randn(100, 10)  # 100个样本，每个10维特征
    y = torch.randint(0, 2, (100,))  # 二分类标签

    # 创建数据集和数据加载器
    class SimpleDataset(Dataset):
        def __init__(self, X: torch.Tensor, y: torch.Tensor):
            self.X = X
            self.y = y

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx]

    dataset = SimpleDataset(X, y)

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 4. 训练模型
    print("\n3. 训练模型:")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练循环
    epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        # 计算平均验证损失和准确率
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证准确率: {accuracy:.2f}%")
        print("-" * 50)


def loss_functions_demo() -> None:
    """
    损失函数示例

    展示PyTorch中常用的损失函数
    """
    print("\n" + "=" * 50)
    print("损失函数示例".center(50))
    print("=" * 50)

    # 1. 回归损失函数
    print("\n1. 回归损失函数:")

    # 准备数据
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([1.1, 1.9, 3.2, 3.7])

    # MSE损失
    mse_loss = nn.MSELoss()
    mse_result = mse_loss(predictions, targets)
    print(f"MSE损失: {mse_result.item():.4f}")

    # MAE损失
    mae_loss = nn.L1Loss()
    mae_result = mae_loss(predictions, targets)
    print(f"MAE损失: {mae_result.item():.4f}")

    # Huber损失
    huber_loss = nn.SmoothL1Loss()
    huber_result = huber_loss(predictions, targets)
    print(f"Huber损失: {huber_result.item():.4f}")

    # 2. 分类损失函数
    print("\n2. 分类损失函数:")

    # 准备数据
    logits = torch.tensor([[0.5, 0.2, 0.3],
                          [0.1, 0.7, 0.2],
                          [0.3, 0.3, 0.4]])
    targets = torch.tensor([0, 1, 2])

    # 交叉熵损失
    ce_loss = nn.CrossEntropyLoss()
    ce_result = ce_loss(logits, targets)
    print(f"交叉熵损失: {ce_result.item():.4f}")

    # 二分类交叉熵损失
    bce_loss = nn.BCEWithLogitsLoss()
    binary_logits = torch.tensor([0.7, -0.3, 0.9])
    binary_targets = torch.tensor([1.0, 0.0, 1.0])
    bce_result = bce_loss(binary_logits, binary_targets)
    print(f"二分类交叉熵损失: {bce_result.item():.4f}")

    print("\n损失函数选择指南:")
    print("回归问题:")
    print("- MSELoss: 标准均方误差，对异常值敏感")
    print("- L1Loss: 平均绝对误差，对异常值不敏感")
    print("- SmoothL1Loss: 结合MSE和MAE的优点")

    print("\n分类问题:")
    print("- CrossEntropyLoss: 多分类问题的标准选择")
    print("- BCEWithLogitsLoss: 二分类问题的标准选择")
    print("- NLLLoss: 当使用log_softmax作为最后一层时使用")


def optimizer_demo() -> None:
    """
    优化器示例

    展示PyTorch中常用的优化器
    """
    print("\n" + "=" * 50)
    print("优化器示例".center(50))
    print("=" * 50)

    # 创建一个简单的模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )

    # 1. SGD优化器
    print("\n1. SGD (随机梯度下降):")
    sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("SGD特点:")
    print("- 最基本的优化算法")
    print("- momentum参数可以帮助逃离局部最小值")
    print("- 学习率需要手动调整")

    # 2. Adam优化器
    print("\n2. Adam:")
    adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    print("Adam特点:")
    print("- 自适应学习率")
    print("- 结合了动量和RMSprop的优点")
    print("- 对超参数不敏感")

    # 3. RMSprop优化器
    print("\n3. RMSprop:")
    rmsprop = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    print("RMSprop特点:")
    print("- 自适应学习率")
    print("- 适合处理非平稳目标")
    print("- 解决AdaGrad学习率递减太快的问题")

    # 4. 学习率调度器示例
    print("\n4. 学习率调度器:")

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 步进式调度器
    step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    print("步进式调度器:")
    print("- 每隔固定步数将学习率乘以gamma")

    # 余弦退火调度器
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    print("\n余弦退火调度器:")
    print("- 学习率按余弦函数从初始值降到最小值再回升")
    print("- 有助于跳出局部最小值")
    
    # ReduceLROnPlateau调度器
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    print("\nReduceLROnPlateau调度器:")
    print("- 当指标停止改善时降低学习率")
    print("- 适合根据验证损失调整学习率")
    
    print("\n优化器选择指南:")
    print("- SGD: 收敛慢但泛化性能好，适合大型模型")
    print("- Adam: 收敛快，适合大多数问题")
    print("- RMSprop: 适合RNN等非平稳问题")
    print("- AdamW: Adam的变体，更好的权重衰减处理")


def training_validation_loop() -> None:
    """
    训练和验证流程
    
    展示完整的神经网络训练和验证流程
    """
    print("\n" + "=" * 50)
    print("训练和验证流程".center(50))
    print("=" * 50)
    
    # 1. 创建一个完整的训练流程
    print("\n1. 完整的训练流程:")
    
    # 定义一个更复杂的模型
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 3)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # 创建模型
    model = MLP()
    
    # 生成一些随机数据
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))
    
    # 创建数据集
    class SimpleDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    dataset = SimpleDataset(X, y)
    
    # 分割数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 训练函数
    def train_epoch(model, dataloader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    
    # 验证函数
    def validate(model, dataloader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss = running_loss / len(dataloader)
        val_acc = 100.0 * correct / total
        return val_loss, val_acc
    
    # 训练循环
    print("\n开始训练...")
    epochs = 10
    best_val_loss = float('inf')
    
    try:
        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
            
            # 验证阶段
            val_loss, val_acc = validate(model, val_loader, criterion)
            
            # 学习率调度
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            
            # 打印统计信息
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"学习率: {current_lr:.6f}")
            print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
            print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 在实际应用中，这里会保存模型
                # torch.save(model.state_dict(), 'best_model.pth')
                print("√ 找到更好的模型!")
            
            print("-" * 50)
            
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        raise
    
    # 测试最佳模型
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"\n测试结果:")
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%")
    
    # 2. 训练技巧和最佳实践
    print("\n2. 训练技巧和最佳实践:")
    
    print("过拟合处理:")
    print("- 使用Dropout层")
    print("- 添加权重衰减(L2正则化)")
    print("- 使用数据增强")
    print("- 早停(Early Stopping)")
    
    print("\n超参数调优:")
    print("- 学习率是最重要的超参数")
    print("- 批量大小影响优化和泛化")
    print("- 使用学习率调度器")
    print("- 考虑使用网格搜索或随机搜索")
    
    print("\n模型保存和加载:")
    print("- 保存整个模型: torch.save(model, 'model.pth')")
    print("- 保存模型参数: torch.save(model.state_dict(), 'model_params.pth')")
    print("- 加载模型参数: model.load_state_dict(torch.load('model_params.pth'))")


def main():
    """运行所有PyTorch神经网络示例"""
    print("\n" + "=" * 80)
    print("PyTorch神经网络教程".center(80))
    print("=" * 80)
    
    # 运行各个部分的示例
    set_matplotlib_chinese()
    linear_layer_basics()
    simple_neural_network()
    loss_functions_demo()
    optimizer_demo()
    training_validation_loop()


if __name__ == "__main__":
    main()