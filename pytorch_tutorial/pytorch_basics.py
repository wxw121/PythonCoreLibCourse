#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch基础知识教程

本模块涵盖PyTorch的基础概念和操作，包括：
1. 张量(Tensor)的创建和操作
2. 自动求导(Autograd)机制
3. GPU加速计算
4. 数据加载和预处理
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional


def tensor_basics() -> None:
    """
    PyTorch张量(Tensor)基础操作

    张量是PyTorch中的核心数据结构，类似于NumPy的ndarray，
    但可以在GPU上运行并支持自动求导
    """
    print("\n" + "=" * 50)
    print("张量(Tensor)基础操作".center(50))
    print("=" * 50)

    # 1. 创建张量的多种方式
    print("\n1. 创建张量的多种方式:")

    # 从Python列表创建
    tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
    print(f"从列表创建: {tensor_from_list}")

    # 从NumPy数组创建
    numpy_array = np.array([1, 2, 3, 4, 5])
    tensor_from_numpy = torch.from_numpy(numpy_array)
    print(f"从NumPy数组创建: {tensor_from_numpy}")
    
    # 创建特定形状的张量
    zeros = torch.zeros(3, 4)  # 3x4的全0张量
    ones = torch.ones(2, 3, 4)  # 2x3x4的全1张量
    rand = torch.rand(2, 3)  # 2x3的随机张量，值在[0,1)之间
    randn = torch.randn(2, 3)  # 2x3的随机张量，服从标准正态分布
    
    print(f"全0张量 (shape={zeros.shape}):\n{zeros}")
    print(f"全1张量 (shape={ones.shape}):\n{ones[0]}")  # 只显示第一个切片以节省空间
    print(f"均匀分布随机张量 (shape={rand.shape}):\n{rand}")
    print(f"正态分布随机张量 (shape={randn.shape}):\n{randn}")
    
    # 创建特定范围的张量
    arange = torch.arange(0, 10, 2)  # 从0到10，步长为2
    linspace = torch.linspace(0, 10, 5)  # 从0到10，均匀分成5份
    
    print(f"arange张量: {arange}")
    print(f"linspace张量: {linspace}")
    
    # 创建单位矩阵
    eye = torch.eye(3)
    print(f"3x3单位矩阵:\n{eye}")
    
    # 2. 张量的属性
    print("\n2. 张量的属性:")
    x = torch.randn(3, 4, 5)
    print(f"张量x的形状: {x.shape}")
    print(f"张量x的维度: {x.dim()}")
    print(f"张量x的元素总数: {x.numel()}")
    print(f"张量x的数据类型: {x.dtype}")
    print(f"张量x的设备: {x.device}")
    
    # 3. 张量的操作
    print("\n3. 张量的基本操作:")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    # 算术运算
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")  # 元素级乘法
    print(f"a / b = {a / b}")
    
    # 矩阵运算
    m1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    m2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    print(f"矩阵乘法 (m1 @ m2):\n{m1 @ m2}")
    print(f"矩阵乘法 (torch.matmul(m1, m2)):\n{torch.matmul(m1, m2)}")
    
    # 4. 张量的索引和切片
    print("\n4. 张量的索引和切片:")
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"张量x:\n{x}")
    print(f"x[1, 2] = {x[1, 2]}")  # 第2行第3列的元素
    print(f"x[0:2, :] =\n{x[0:2, :]}")  # 前2行的所有列
    print(f"x[:, 1:3] =\n{x[:, 1:3]}")  # 所有行的第2和第3列
    
    # 5. 张量的形状操作
    print("\n5. 张量的形状操作:")
    x = torch.randn(2, 3)
    print(f"原始张量x (shape={x.shape}):\n{x}")
    
    # 改变形状
    x_reshaped = x.reshape(3, 2)
    print(f"reshape后 (shape={x_reshaped.shape}):\n{x_reshaped}")
    
    # 转置
    x_t = x.t()
    print(f"转置后 (shape={x_t.shape}):\n{x_t}")
    
    # 添加维度
    x_unsqueezed = x.unsqueeze(0)  # 在第0维添加一个维度
    print(f"unsqueeze后 (shape={x_unsqueezed.shape}):\n{x_unsqueezed}")
    
    # 压缩维度
    x_squeezed = x_unsqueezed.squeeze(0)  # 压缩第0维
    print(f"squeeze后 (shape={x_squeezed.shape}):\n{x_squeezed}")
    
    # 6. 张量的连接和堆叠
    print("\n6. 张量的连接和堆叠:")
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])
    
    # 沿着第0维连接（垂直方向）
    c = torch.cat([a, b], dim=0)
    print(f"垂直连接 (shape={c.shape}):\n{c}")
    
    # 沿着第1维连接（水平方向）
    d = torch.cat([a, b], dim=1)
    print(f"水平连接 (shape={d.shape}):\n{d}")
    
    # 堆叠（增加一个新维度）
    e = torch.stack([a, b])
    print(f"堆叠 (shape={e.shape}):\n{e}")


def autograd_basics() -> None:
    """
    PyTorch自动求导(Autograd)基础
    
    自动求导是PyTorch的核心功能之一，它能够自动计算神经网络中的梯度，
    这对于实现反向传播算法至关重要
    """
    print("\n" + "=" * 50)
    print("自动求导(Autograd)基础".center(50))
    print("=" * 50)
    
    # 1. 创建需要梯度的张量
    print("\n1. 创建需要梯度的张量:")
    x = torch.tensor(2.0, requires_grad=True)
    print(f"x = {x}, requires_grad = {x.requires_grad}")
    
    # 2. 构建计算图
    print("\n2. 构建计算图:")
    y = x ** 2 + 3 * x + 1
    print(f"y = x^2 + 3x + 1 = {y}")
    
    # 3. 反向传播计算梯度
    print("\n3. 反向传播计算梯度:")
    y.backward()
    print(f"dy/dx = 2x + 3 = {x.grad}")  # 当x=2时，梯度应为2*2+3=7
    
    # 4. 复杂例子：多变量函数
    print("\n4. 多变量函数的梯度计算:")
    # 重置计算图
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.sum(x ** 2)  # y = x[0]^2 + x[1]^2 + x[2]^2
    print(f"x = {x}")
    print(f"y = sum(x^2) = {y}")
    
    # 计算梯度
    y.backward()
    print(f"梯度 dy/dx = 2*x = {x.grad}")  # 应为[2, 4, 6]
    
    # 5. 梯度累积和梯度清零
    print("\n5. 梯度累积和梯度清零:")
    x = torch.tensor(2.0, requires_grad=True)
    
    # 第一次计算
    y = x ** 2
    y.backward()
    print(f"第一次计算后的梯度: {x.grad}")  # 应为4
    
    # 梯度会累积，如果不清零
    y = x ** 2
    y.backward()
    print(f"不清零时的梯度（累积）: {x.grad}")  # 应为8 (4+4)
    
    # 清零梯度
    x.grad.zero_()
    y = x ** 2
    y.backward()
    print(f"清零后重新计算的梯度: {x.grad}")  # 应为4
    
    # 6. 使用with torch.no_grad()暂停梯度计算
    print("\n6. 暂停梯度计算:")
    x = torch.tensor(2.0, requires_grad=True)
    
    # 正常情况下会计算梯度
    y = x ** 2
    print(f"y需要梯度: {y.requires_grad}")
    
    # 使用no_grad()暂停梯度计算
    with torch.no_grad():
        z = x ** 3
    print(f"z需要梯度: {z.requires_grad}")
    
    # 7. 实际应用：简单线性回归
    print("\n7. 实际应用：简单线性回归")
    
    # 生成一些带有噪声的线性数据
    x_data = torch.linspace(-5, 5, 100)
    print(f"x_data: {x_data}")
    y_data = 3 * x_data + 2 + 0.5 * torch.randn(100)
    
    # 初始化参数
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    
    # 学习率
    learning_rate = 0.01
    
    # 训练循环
    print("训练线性回归模型...")
    for epoch in range(100):
        # 前向传播
        y_pred = w * x_data + b
        
        # 计算损失
        loss = torch.mean((y_pred - y_data) ** 2)
        
        # 反向传播
        loss.backward()
        
        # 更新参数（手动，不使用优化器）
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            
            # 清零梯度
            w.grad.zero_()
            b.grad.zero_()
        
        # 每20个epoch打印一次进度
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")
    
    print(f"训练结束。真实参数: w=3, b=2; 学习到的参数: w={w.item():.4f}, b={b.item():.4f}")


def gpu_acceleration() -> None:
    """
    PyTorch GPU加速计算
    
    PyTorch可以无缝地将计算从CPU转移到GPU，以加速深度学习模型的训练
    """
    print("\n" + "=" * 50)
    print("GPU加速计算".center(50))
    print("=" * 50)
    
    # 1. 检查GPU是否可用
    print("\n1. 检查GPU是否可用:")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        print(f"GPU可用! 设备数量: {device_count}")
        print(f"当前设备索引: {current_device}")
        print(f"当前设备名称: {device_name}")
        
        # 设置默认设备
        device = torch.device("cuda")
    else:
        print("GPU不可用，将使用CPU")
        device = torch.device("cpu")
    
    # 2. 创建张量并移动到指定设备
    print("\n2. 创建张量并移动到指定设备:")
    
    # 在CPU上创建张量
    x_cpu = torch.randn(3, 4)
    print(f"x_cpu设备: {x_cpu.device}")
    
    # 将张量移动到指定设备
    x_device = x_cpu.to(device)
    print(f"x_device设备: {x_device.device}")
    
    # 直接在指定设备上创建张量
    y_device = torch.randn(3, 4, device=device)
    print(f"y_device设备: {y_device.device}")
    
    # 3. 在不同设备上的运算
    print("\n3. 在不同设备上的运算:")
    try:
        # 这将引发错误，因为张量在不同设备上
        z = x_cpu + y_device
    except RuntimeError as e:
        print("错误：不能直接在不同设备上的张量之间进行运算")
        print("正确做法是确保两个张量在同一个设备上")
    
    # 正确的做法
    z = x_device + y_device
    print(f"z设备: {z.device}")
    
    # 4. 模型和GPU
    print("\n4. 模型和GPU示例:")
    
    # 创建一个简单的神经网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
            
        def forward(self, x):
            return self.fc(x)
    
    # 创建模型实例
    model = SimpleNet()
    print(f"模型默认设备: {next(model.parameters()).device}")
    
    # 将模型移动到指定设备
    model = model.to(device)
    print(f"移动后的模型设备: {next(model.parameters()).device}")
    
    # 5. 性能对比（如果有GPU）
    if torch.cuda.is_available():
        print("\n5. CPU vs GPU性能对比:")
        
        # 创建大矩阵
        size = 2000
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        # CPU计时
        start_time = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU矩阵乘法用时: {cpu_time:.4f}秒")
        
        # GPU计时
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        
        # 预热GPU
        _ = torch.mm(a_gpu, b_gpu)
        
        start_time = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        gpu_time = time.time() - start_time
        print(f"GPU矩阵乘法用时: {gpu_time:.4f}秒")
        print(f"GPU加速比: {cpu_time/gpu_time:.2f}x")


def data_loading_basics() -> None:
    """
    PyTorch数据加载和预处理基础
    
    展示如何使用PyTorch的数据加载工具来处理数据集
    """
    print("\n" + "=" * 50)
    print("数据加载和预处理基础".center(50))
    print("=" * 50)
    
    # 1. 创建自定义数据集
    print("\n1. 创建自定义数据集:")
    
    class CustomDataset(Dataset):
        """自定义数据集示例"""
        
        def __init__(self, size: int = 1000):
            """
            初始化数据集
            
            Args:
                size: 数据集大小
            """
            self.size = size
            # 生成随机数据
            self.data = torch.randn(size, 10)  # 10维特征
            self.labels = torch.randint(0, 2, (size,))  # 二分类标签
            
        def __len__(self) -> int:
            """返回数据集大小"""
            return self.size
            
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            获取一个数据样本
            
            Args:
                idx: 样本索引
                
            Returns:
                特征和标签的元组
            """
            return self.data[idx], self.labels[idx]
    
    # 创建数据集实例
    dataset = CustomDataset()
    print(f"数据集大小: {len(dataset)}")
    print(f"第一个样本: {dataset[0]}")
    
    # 2. 使用DataLoader加载数据
    print("\n2. 使用DataLoader加载数据:")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Windows下多进程可能有问题，设为0
    )
    
    print(f"DataLoader批次数: {len(dataloader)}")
    
    # 获取一个批次的数据
    features, labels = next(iter(dataloader))
    print(f"一个批次的形状: 特征{features.shape}, 标签{labels.shape}")
    
    # 3. 数据转换和预处理
    print("\n3. 数据转换和预处理示例:")
    
    from torchvision import transforms
    
    # 创建转换流水线
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化
    ])
    
    print("常用的转换操作包括:")
    print("- ToTensor(): 将PIL图像或NumPy数组转换为张量")
    print("- Normalize(): 标准化张量")
    print("- Resize(): 调整图像大小")
    print("- RandomCrop(): 随机裁剪")
    print("- RandomHorizontalFlip(): 随机水平翻转")
    print("- ColorJitter(): 随机调整亮度、对比度等")
    
    # 4. 数据采样器示例
    print("\n4. 数据采样器示例:")
    
    from torch.utils.data import WeightedRandomSampler
    
    # 创建带权重的采样器
    weights = torch.ones(len(dataset))  # 所有样本权重相等
    sampler = WeightedRandomSampler(weights, len(dataset))
    
    # 使用采样器的DataLoader
    weighted_dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0
    )
    
    print("采样器类型:")
    print("- RandomSampler: 随机采样")
    print("- SequentialSampler: 顺序采样")
    print("- WeightedRandomSampler: 带权重的随机采样")
    print("- BatchSampler: 批次采样")


def main():
    """运行所有PyTorch基础示例"""
    print("\n" + "=" * 80)
    print("PyTorch基础教程".center(80))
    print("=" * 80)
    
    # 运行各个部分的示例
    tensor_basics()
    autograd_basics()
    gpu_acceleration()
    data_loading_basics()


if __name__ == "__main__":
    main()