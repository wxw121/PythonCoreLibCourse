#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch计算机视觉教程

本模块涵盖PyTorch在计算机视觉领域的应用，包括：
1. 图像数据处理
2. 卷积神经网络(CNN)基础
3. 图像分类实战
4. 数据增强技术
5. 迁移学习
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image


def image_data_processing() -> None:
    """
    图像数据处理基础
    
    展示如何使用PyTorch和torchvision处理图像数据
    """
    print("\n" + "=" * 50)
    print("图像数据处理基础".center(50))
    print("=" * 50)
    
    # 1. 图像数据加载和转换
    print("\n1. 图像数据加载和转换:")
    
    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                             std=[0.229, 0.224, 0.225])
    ])
    
    print("常用图像转换操作:")
    print("- Resize: 调整图像大小")
    print("- CenterCrop/RandomCrop: 中心/随机裁剪")
    print("- ToTensor: 将PIL图像转换为张量")
    print("- Normalize: 标准化图像")
    print("- RandomHorizontalFlip: 随机水平翻转")
    print("- ColorJitter: 随机调整亮度、对比度等")
    
    # 2. 使用内置数据集
    print("\n2. 使用内置数据集:")
    
    print("torchvision提供的常用数据集:")
    print("- MNIST: 手写数字")
    print("- CIFAR-10/CIFAR-100: 小型彩色图像")
    print("- ImageNet: 大规模图像分类数据集")
    print("- COCO: 目标检测、分割和字幕数据集")
    
    # 示例：如何加载CIFAR-10数据集
    print("\n加载CIFAR-10数据集示例代码:")
    print("""
    # 训练集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 测试集
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    """)
    
    # 3. 自定义图像数据集
    print("\n3. 自定义图像数据集:")
    
    class CustomImageDataset(Dataset):
        """自定义图像数据集示例"""
        
        def __init__(self, img_dir: str, transform=None):
            """
            初始化数据集
            
            Args:
                img_dir: 图像目录路径
                transform: 图像转换操作
            """
            self.img_dir = img_dir
            self.transform = transform
            self.img_files = [f for f in os.listdir(img_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
            
        def __len__(self) -> int:
            """返回数据集大小"""
            return len(self.img_files)
            
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
            """
            获取一个数据样本
            
            Args:
                idx: 样本索引
                
            Returns:
                图像张量和图像文件名
            """
            img_path = os.path.join(self.img_dir, self.img_files[idx])
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, self.img_files[idx]
    
    print("自定义图像数据集的关键步骤:")
    print("1. 继承torch.utils.data.Dataset类")
    print("2. 实现__len__方法返回数据集大小")
    print("3. 实现__getitem__方法加载和处理单个样本")
    print("4. 使用transform参数应用图像转换")
    
    # 4. 图像批处理和可视化
    print("\n4. 图像批处理和可视化:")
    
    # 创建一些随机图像数据
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    
    # 创建随机批次
    batch = torch.randn(batch_size, channels, height, width)
    
    print(f"批次形状: {batch.shape}")
    print("PyTorch中图像张量的维度顺序: [批次大小, 通道数, 高度, 宽度]")
    
    # 图像可视化函数
    def show_images(images):
        """显示一批图像"""
        # 将图像从[B,C,H,W]转换为[B,H,W,C]用于显示
        images = images.permute(0, 2, 3, 1).numpy()
        
        # 反标准化（如果需要）
        # images = images * std + mean
        
        # 裁剪到[0,1]范围
        images = np.clip(images, 0, 1)
        
        # 创建子图
        fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
        for i, img in enumerate(images):
            if len(images) == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.close()
    
    print("图像可视化的关键步骤:")
    print("1. 将张量从[B,C,H,W]转换为[B,H,W,C]")
    print("2. 反标准化图像（如果之前进行了标准化）")
    print("3. 裁剪像素值到合适范围")
    print("4. 使用matplotlib显示图像")


def cnn_basics() -> None:
    """
    卷积神经网络(CNN)基础
    
    展示卷积神经网络的基本组件和构建方法
    """
    print("\n" + "=" * 50)
    print("卷积神经网络(CNN)基础".center(50))
    print("=" * 50)
    
    # 1. CNN的基本组件
    print("\n1. CNN的基本组件:")
    
    # 卷积层
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    print("卷积层 (Conv2d):")
    print("- in_channels: 输入通道数")
    print("- out_channels: 输出通道数（卷积核数量）")
    print("- kernel_size: 卷积核大小")
    print("- stride: 步长")
    print("- padding: 填充")
    
    # 池化层
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    print("\n池化层:")
    print("- MaxPool2d: 最大池化，提取最显著特征")
    print("- AvgPool2d: 平均池化，保留更多背景信息")
    
    # 批标准化
    batch_norm = nn.BatchNorm2d(16)
    print("\n批标准化 (BatchNorm2d):")
    print("- 加速训练收敛")
    print("- 允许使用更高的学习率")
    print("- 减少对初始化的依赖")
    print("- 具有轻微的正则化效果")
    
    # 激活函数
    print("\nCNN中常用的激活函数:")
    print("- ReLU: 最常用的激活函数")
    print("- Leaky ReLU: 解决'死亡ReLU'问题")
    print("- ELU: 指数线性单元")
    
    # 2. 构建一个简单的CNN
    print("\n2. 构建一个简单的CNN:")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            # 第一个卷积块
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(2)
            
            # 第二个卷积块
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(2)
            
            # 第三个卷积块
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d(2)
            
            # 全连接层
            self.fc1 = nn.Linear(64 * 4 * 4, 128)
            self.fc2 = nn.Linear(128, 10)
            
        def forward(self, x):
            # 第一个卷积块
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            
            # 第二个卷积块
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            
            # 第三个卷积块
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            
            # 展平
            x = x.view(x.size(0), -1)
            
            # 全连接层
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            
            return x
    
    # 创建模型实例
    model = SimpleCNN()
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    # 3. 卷积层的可视化理解
    print("\n3. 卷积层的可视化理解:")
    
    print("卷积操作的工作原理:")
    print("- 卷积核在输入上滑动，计算点积")
    print("- 不同的卷积核学习不同的特征")
    print("- 浅层卷积学习边缘、纹理等低级特征")
    print("- 深层卷积学习形状、物体部分等高级特征")
    
    # 4. CNN架构设计模式
    print("\n4. CNN架构设计模式:")
    
    print("常见的CNN架构模式:")
    print("- 标准CNN: 卷积层+池化层+全连接层")
    print("- VGG风格: 多个3x3卷积层堆叠")
    print("- ResNet风格: 添加残差连接")
    print("- Inception风格: 并行使用不同大小的卷积")
    print("- DenseNet风格: 密集连接所有层")


def image_classification() -> None:
    """
    图像分类实战

    展示如何使用PyTorch构建和训练图像分类模型
    """
    print("\n" + "=" * 50)
    print("图像分类实战".center(50))
    print("=" * 50)

    # 1. 构建图像分类模型
    print("\n1. 构建图像分类模型:")

    class ConvBlock(nn.Module):
        """卷积块：卷积+批标准化+ReLU+池化"""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x)
            return x

    class ImageClassifier(nn.Module):
        """图像分类模型"""

        def __init__(self, num_classes=10):
            super().__init__()

            # 卷积层
            self.conv_layers = nn.Sequential(
                ConvBlock(3, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256)
            )

            # 全连接层
            self.fc_layers = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 2 * 2, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            # 输入图像应为[batch_size, 3, 32, 32]
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x
    
    # 创建模型实例
    model = ImageClassifier(num_classes=10)
    print("图像分类模型结构:")
    print(model)
    
    # 2. 训练图像分类模型
    print("\n2. 训练图像分类模型:")
    
    print("图像分类模型训练流程:")
    print("1. 准备数据和数据加载器")
    print("2. 定义损失函数和优化器")
    print("3. 训练循环:")
    print("   - 前向传播")
    print("   - 计算损失")
    print("   - 反向传播")
    print("   - 更新参数")
    print("4. 验证和测试")
    
    # 训练函数示例
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        """
        训练图像分类模型
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            num_epochs: 训练轮数
        """
        # 检查是否有GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 计算训练指标
            train_loss = train_loss / len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    # 前向传播
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    # 统计
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # 计算验证指标
            val_loss = val_loss / len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            # 打印统计信息
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print("-" * 50)
    
    # 3. 模型评估和预测
    print("\n3. 模型评估和预测:")
    
    # 评估函数
    def evaluate_model(model, test_loader):
        """
        评估模型性能
        
        Args:
            model: 模型
            test_loader: 测试数据加载器
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        test_correct = 0
        test_total = 0
        
        # 用于计算每个类别的准确率
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                # 总体准确率
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # 每个类别的准确率
                for i in range(labels.size(0)):
                    label = labels[i]
                    pred = predicted[i]
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
        
        # 计算总体准确率
        test_acc = 100.0 * test_correct / test_total
        print(f"测试准确率: {test_acc:.2f}%")
        
        # 计算每个类别的准确率
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                print(f"类别 {i} 准确率: {class_acc:.2f}%")
    
    # 预测函数
    def predict_image(model, image_tensor):
        """
        预测单个图像的类别
        
        Args:
            model: 模型
            image_tensor: 图像张量 [1, C, H, W]
        
        Returns:
            预测的类别索引和概率
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # 获取最高概率的类别
            prob, predicted = torch.max(probabilities, 1)
            
        return predicted.item(), prob.item()


def data_augmentation() -> None:
    """
    数据增强技术
    
    展示如何使用数据增强来提高模型性能
    """
    print("\n" + "=" * 50)
    print("数据增强技术".center(50))
    print("=" * 50)
    
    # 1. 常用数据增强技术
    print("\n1. 常用数据增强技术:")
    
    # 创建一个综合的数据增强转换
    transform_train = transforms.Compose([
        # 随机裁剪
        transforms.RandomCrop(32, padding=4),
        
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        
        # 随机旋转
        transforms.RandomRotation(15),
        
        # 随机调整亮度、对比度、饱和度和色调
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        
        # 随机擦除部分图像
        transforms.RandomErasing(p=0.2),
        
        # 转换为张量
        transforms.ToTensor(),
        
        # 标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("数据增强的常用技术:")
    print("- RandomCrop: 随机裁剪图像")
    print("- RandomHorizontalFlip: 随机水平翻转")
    print("- RandomRotation: 随机旋转")
    print("- ColorJitter: 随机调整亮度、对比度等")
    print("- RandomErasing: 随机擦除部分图像")
    print("- RandomResizedCrop: 随机裁剪并调整大小")
    print("- RandomAffine: 随机仿射变换")
    
    # 2. 数据增强的好处
    print("\n2. 数据增强的好处:")
    print("- 增加训练数据的多样性")
    print("- 减少过拟合")
    print("- 提高模型的泛化能力")
    print("- 提高模型对各种变化的鲁棒性")
    
    # 3. 自定义数据增强
    print("\n3. 自定义数据增强:")
    
    class GaussianNoise:
        """添加高斯噪声的自定义转换"""
        
        def __init__(self, mean=0., std=1.):
            self.mean = mean
            self.std = std
            
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    # 使用自定义转换
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        GaussianNoise(0., 0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("创建自定义数据增强的步骤:")
    print("1. 定义一个类，实现__call__方法")
    print("2. 在__call__方法中实现转换逻辑")
    print("3. 将自定义转换添加到transforms.Compose中")
    
    # 4. 在线数据增强 vs. 离线数据增强
    print("\n4. 在线数据增强 vs. 离线数据增强:")
    print("在线数据增强:")
    print("- 在训练过程中实时应用")
    print("- 每个epoch看到不同的增强版本")
    print("- 不需要额外存储空间")
    print("- 可能会增加训练时间")
    
    print("\n离线数据增强:")
    print("- 预先生成增强数据并保存")
    print("- 训练速度更快")
    print("- 需要额外存储空间")
    print("- 增强的多样性有限")


def transfer_learning() -> None:
    """
    迁移学习
    
    展示如何使用预训练模型进行迁移学习
    """
    print("\n" + "=" * 50)
    print("迁移学习".center(50))
    print("=" * 50)
    
    # 1. 迁移学习基础
    print("\n1. 迁移学习基础:")
    print("迁移学习是利用在一个任务上训练好的模型来提高另一个相关任务的学习效率的技术")
    print("优点:")
    print("- 减少训练时间")
    print("- 需要更少的训练数据")
    print("- 通常能获得更好的性能")
    print("- 适用于数据有限的情况")
    
    # 2. 加载预训练模型
    print("\n2. 加载预训练模型:")
    
    # 加载预训练的ResNet-18
    resnet = models.resnet18(pretrained=True)
    print("预训练ResNet-18模型结构:")
    print(resnet)
    
    print("\ntorchvision提供的预训练模型:")
    print("- ResNet系列: resnet18, resnet34, resnet50, ...")
    print("- VGG系列: vgg11, vgg13, vgg16, vgg19, ...")
    print("- DenseNet系列: densenet121, densenet169, ...")
    print("- Inception系列: inception_v3")
    print("- MobileNet系列: mobilenet_v2, mobilenet_v3_small, ...")
    print("- EfficientNet系列: efficientnet_b0, efficientnet_b1, ...")
    
    # 3. 特征提取
    print("\n3. 特征提取:")
    
    # 冻结所有预训练层
    for param in resnet.parameters():
        param.requires_grad = False
    
    # 替换最后的全连接层
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, 10)  # 10个类别
    
    print("特征提取步骤:")
    print("1. 加载预训练模型")
    print("2. 冻结预训练层的参数")
    print("3. 替换分类层")
    print("4. 只训练新的分类层")
    
    # 4. 微调
    print("\n4. 微调:")
    
    # 加载预训练模型
    resnet_finetune = models.resnet18(pretrained=True)
    
    # 解冻部分层
    # 冻结前面的层
    for name, param in resnet_finetune.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 替换最后的全连接层
    num_features = resnet_finetune.fc.in_features
    resnet_finetune.fc = nn.Linear(num_features, 10)
    
    print("微调步骤:")
    print("1. 加载预训练模型")
    print("2. 冻结部分预训练层")
    print("3. 替换分类层")
    print("4. 使用较小的学习率训练")
    
    # 5. 迁移学习最佳实践
    print("\n5. 迁移学习最佳实践:")
    print("- 当目标数据集较小时，只训练分类层")
    print("- 当目标数据集较大时，可以微调更多层")
    print("- 使用较小的学习率进行微调")
    print("- 考虑使用不同的学习率：预训练层较小，新层较大")
    print("- 使用适当的数据增强")
    
    # 6. 迁移学习示例代码
    print("\n6. 迁移学习示例代码:")
    
    def train_transfer_learning_model():
        """迁移学习训练示例"""
        # 加载预训练模型
        model = models.resnet18(pretrained=True)
        
        # 冻结所有预训练层
        for param in model.parameters():
            param.requires_grad = False
        
        # 替换分类层
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        
        # 只优化分类层的参数
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        
        # 训练循环
        # ...
        
        return model
    
    print("完整的迁移学习流程包括:")
    print("1. 准备数据集和数据加载器")
    print("2. 加载预训练模型并修改")
    print("3. 定义损失函数和优化器")
    print("4. 训练和验证")
    print("5. 评估最终性能")


def main():
    """运行所有PyTorch计算机视觉示例"""
    print("\n" + "=" * 80)
    print("PyTorch计算机视觉教程".center(80))
    print("=" * 80)
    
    # 运行各个部分的示例
    image_data_processing()
    cnn_basics()
    image_classification()
    data_augmentation()
    transfer_learning()


if __name__ == "__main__":
    main()