"""
Scikit-learn 基础知识

本模块介绍scikit-learn的基本概念、数据集加载和操作、模型训练和预测的基本流程以及常用工具和辅助函数。
这是scikit-learn教程的第一部分，为后续学习打下基础。
"""

import os
# 导入matplotlib配置（必须在任何其他matplotlib导入之前）
from matplotlib_config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import set_matplotlib_chinese
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def introduction_to_sklearn():
    """
    介绍scikit-learn库及其主要组件
    """
    print("=" * 50)
    print("Scikit-learn 简介".center(50))
    print("=" * 50)

    print("\nScikit-learn是Python中最流行的机器学习库之一，提供了：")
    print("1. 简单且高效的工具，用于数据挖掘和数据分析")
    print("2. 可访问性强，适合各种背景的人使用")
    print("3. 构建在NumPy、SciPy和Matplotlib之上")
    print("4. 开源、商业可用（BSD许可证）")

    print("\nScikit-learn的主要组件：")
    print("- 分类：识别某个对象属于哪个类别")
    print("- 回归：预测与对象相关的连续值属性")
    print("- 聚类：将相似对象自动分组")
    print("- 降维：减少要考虑的随机变量的数量")
    print("- 模型选择：比较、验证和选择参数和模型")
    print("- 预处理：特征提取和归一化")

    print("\nScikit-learn的设计原则：")
    print("- 一致性：所有对象共享一个简单的接口")
    print("- 检查：所有参数值都经过验证")
    print("- 限制对象层次结构：只使用Python内置类型")
    print("- 组合：许多算法可以组合使用")
    print("- 合理默认值：使模型无需大量调参即可工作")


def loading_datasets():
    """
    演示如何加载和探索scikit-learn内置数据集
    """
    print("\n" + "=" * 50)
    print("数据集加载和探索".center(50))
    print("=" * 50)

    # 1. 加载内置数据集
    print("\n1. 加载内置数据集")

    # 鸢尾花数据集
    iris = datasets.load_iris()
    print("\n鸢尾花数据集:")
    print(f"数据形状: {iris.data.shape}")
    print(f"目标形状: {iris.target.shape}")
    print(f"特征名称: {iris.feature_names}")
    print(f"目标名称: {iris.target_names}")

    # 乳腺癌数据集
    cancer = datasets.load_breast_cancer()
    print("\n乳腺癌数据集:")
    print(f"数据形状: {cancer.data.shape}")
    print(f"目标形状: {cancer.target.shape}")
    print(f"特征数量: {len(cancer.feature_names)}")
    print(f"目标类别: {np.unique(cancer.target)}")

    # 波士顿房价数据集（如果可用）
    try:
        boston = datasets.load_boston()
        print("\n波士顿房价数据集:")
        print(f"数据形状: {boston.data.shape}")
        print(f"目标形状: {boston.target.shape}")
    except:
        # 如果波士顿数据集不可用，使用加州房价数据集
        california = datasets.fetch_california_housing()
        print("\n加州房价数据集:")
        print(f"数据形状: {california.data.shape}")
        print(f"目标形状: {california.target.shape}")

    # 2. 生成合成数据集
    print("\n2. 生成合成数据集")

    # 生成分类数据集
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    print("\n合成分类数据集:")
    print(f"数据形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    print(f"类别分布: {np.bincount(y)}")

    # 生成回归数据集
    X_reg, y_reg = datasets.make_regression(
        n_samples=1000,
        n_features=10,
        noise=20,
        random_state=42
    )
    print("\n合成回归数据集:")
    print(f"数据形状: {X_reg.shape}")
    print(f"目标形状: {y_reg.shape}")
    print(f"目标范围: [{y_reg.min():.2f}, {y_reg.max():.2f}]")

    # 3. 数据可视化
    print("\n3. 数据可视化")

    # 使用鸢尾花数据集进行可视化
    plt.figure(figsize=(12, 5))

    # 散点图
    plt.subplot(1, 2, 1)
    plt.scatter(
        iris.data[:, 0], iris.data[:, 1],
        c=iris.target, cmap='viridis',
        edgecolor='k', s=50
    )
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('鸢尾花数据集 - 萼片长度 vs 萼片宽度')

    # 箱线图
    plt.subplot(1, 2, 2)
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = pd.Categorical.from_codes(
        iris.target, iris.target_names
    )
    sns.boxplot(x='species', y=iris.feature_names[0], data=df)
    plt.title('鸢尾花数据集 - 萼片长度分布')

    plt.tight_layout()
    
    # 确保images目录存在
    os.makedirs("images", exist_ok=True)
    
    plt.savefig("images/dataset_visualization.png")
    plt.close()

    print("\n数据可视化已保存为 'images/dataset_visualization.png'")

    return cancer  # 返回乳腺癌数据集用于后续示例


def data_preprocessing(dataset):
    """
    演示基本的数据预处理步骤
    """
    print("\n" + "=" * 50)
    print("数据预处理".center(50))
    print("=" * 50)

    X = dataset.data
    y = dataset.target

    print("\n原始数据概览:")
    print(f"特征形状: {X.shape}")
    print(f"目标形状: {y.shape}")
    print(f"特征范围: [{X.min():.2f}, {X.max():.2f}]")

    # 1. 划分训练集和测试集
    print("\n1. 划分训练集和测试集")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"训练集形状: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"测试集形状: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"训练集类别分布: {np.bincount(y_train)}")
    print(f"测试集类别分布: {np.bincount(y_test)}")

    # 2. 特征缩放
    print("\n2. 特征缩放")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("缩放前:")
    print(f"训练集均值: {X_train.mean(axis=0)[:5]}...")
    print(f"训练集标准差: {X_train.std(axis=0)[:5]}...")

    print("\n缩放后:")
    print(f"训练集均值: {X_train_scaled.mean(axis=0)[:5]}...")
    print(f"训练集标准差: {X_train_scaled.std(axis=0)[:5]}...")

    # 3. 特征相关性分析
    print("\n3. 特征相关性分析")

    # 创建DataFrame以便于计算相关性
    df = pd.DataFrame(X, columns=dataset.feature_names)

    # 计算相关性矩阵
    corr_matrix = df.corr()

    # 找出高度相关的特征对
    high_corr_threshold = 0.8
    high_corr_features = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
                high_corr_features.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j],
                     corr_matrix.iloc[i, j])
                )

    print(f"高度相关特征对 (|r| > {high_corr_threshold}):")
    for feat1, feat2, corr in high_corr_features[:5]:
        print(f"  {feat1} -- {feat2}: {corr:.4f}")

    if len(high_corr_features) > 5:
        print(f"  ...以及{len(high_corr_features) - 5}个其他特征对")

    # 可视化相关性矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix.iloc[:10, :10],  # 只显示前10个特征以便于可视化
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        fmt='.2f'
    )
    plt.title('特征相关性矩阵 (前10个特征)')
    plt.tight_layout()
    plt.savefig("images/feature_correlation.png")
    plt.close()

    print("\n特征相关性矩阵已保存为 'images/feature_correlation.png'")

    return X_train_scaled, X_test_scaled, y_train, y_test


def basic_model_training(X_train, X_test, y_train, y_test):
    """
    演示基本的模型训练和评估流程
    """
    print("\n" + "=" * 50)
    print("模型训练和评估".center(50))
    print("=" * 50)

    # 创建不同的分类器
    classifiers = {
        "逻辑回归": LogisticRegression(random_state=42, max_iter=1000),
        "决策树": DecisionTreeClassifier(random_state=42),
        "K近邻": KNeighborsClassifier(n_neighbors=5),
        "支持向量机": SVC(random_state=42),
        "随机森林": RandomForestClassifier(random_state=42)
    }

    # 训练和评估每个分类器
    results = {}

    for name, clf in classifiers.items():
        print(f"\n训练 {name}...")

        # 训练模型
        clf.fit(X_train, y_train)

        # 在测试集上预测
        y_pred = clf.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"{name} 准确率: {accuracy:.4f}")

        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

    # 可视化模型比较
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = list(results.values())

    # 按准确率排序
    sorted_idx = np.argsort(accuracies)
    models = [models[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]

    plt.barh(models, accuracies, color='skyblue')
    plt.xlabel('准确率')
    plt.title('不同分类器的准确率比较')
    plt.xlim(0, 1)

    # 在条形上添加准确率值
    for i, v in enumerate(accuracies):
        plt.text(v + 0.01, i, f"{v:.4f}", va='center')

    plt.tight_layout()
    plt.savefig("images/model_comparison.png")
    plt.close()

    print("\n模型比较图已保存为 'images/model_comparison.png'")

    # 返回性能最好的模型
    best_model_name = max(results, key=results.get)
    best_model = classifiers[best_model_name]

    print(f"\n性能最好的模型是 {best_model_name}，准确率为 {results[best_model_name]:.4f}")

    return best_model


def model_prediction_workflow(model, X_test, y_test):
    """
    演示完整的模型预测和结果分析流程
    """
    print("\n" + "=" * 50)
    print("模型预测和结果分析".center(50))
    print("=" * 50)
    
    # 1. 获取预测结果
    print("\n1. 获取预测结果")
    y_pred = model.predict(X_test)
    
    # 如果模型支持概率预测
    try:
        y_prob = model.predict_proba(X_test)
        has_probabilities = True
        print("模型支持概率预测")
    except:
        has_probabilities = False
        print("模型不支持概率预测")
    
    # 2. 混淆矩阵分析
    print("\n2. 混淆矩阵分析")
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    
    plt.savefig("images/confusion_matrix.png")
    plt.close()
    
    print("\n混淆矩阵已保存为 'images/confusion_matrix.png'")
    
    # 3. 错误分析
    print("\n3. 错误分析")
    incorrect_indices = np.where(y_test != y_pred)[0]
    n_errors = len(incorrect_indices)
    
    print(f"错误预测数量: {n_errors} (占测试集的 {n_errors/len(y_test)*100:.2f}%)")
    
    if has_probabilities:
        # 分析错误预测的置信度
        error_probs = np.max(y_prob[incorrect_indices], axis=1)
        avg_error_confidence = np.mean(error_probs)
        print(f"错误预测的平均置信度: {avg_error_confidence:.4f}")
        
        # 高置信度错误
        high_conf_errors = incorrect_indices[error_probs > 0.8]
        print(f"高置信度错误数量 (>0.8): {len(high_conf_errors)}")
    
    # 4. 决策边界可视化（仅适用于二维数据）
    # 这里我们只是提供一个示例，实际上我们的数据可能是高维的
    print("\n4. 决策边界可视化")
    print("注意: 此示例仅适用于二维数据，我们的数据可能是高维的")
    print("在实际应用中，可以使用降维技术（如PCA）将高维数据投影到2D进行可视化")


def scikit_learn_workflow():
    """
    演示完整的scikit-learn工作流程
    """
    print("\n" + "=" * 50)
    print("Scikit-learn 工作流程".center(50))
    print("=" * 50)
    
    print("\nScikit-learn的典型工作流程包括以下步骤：")
    print("1. 数据加载和探索")
    print("2. 数据预处理")
    print("3. 模型训练")
    print("4. 模型评估")
    print("5. 模型调优")
    print("6. 模型部署和预测")
    
    # 1. 数据加载和探索
    print("\n步骤1: 数据加载和探索")
    dataset = loading_datasets()
    
    # 2. 数据预处理
    print("\n步骤2: 数据预处理")
    X_train, X_test, y_train, y_test = data_preprocessing(dataset)
    
    # 3. 模型训练
    print("\n步骤3: 模型训练")
    best_model = basic_model_training(X_train, X_test, y_train, y_test)
    
    # 4. 模型评估和预测
    print("\n步骤4: 模型评估和预测")
    model_prediction_workflow(best_model, X_test, y_test)
    
    print("\n" + "=" * 50)
    print("工作流程完成".center(50))
    print("=" * 50)


def main():
    """主函数，运行所有示例"""
    # 设置matplotlib显示中文
    set_matplotlib_chinese()
    
    introduction_to_sklearn()
    scikit_learn_workflow()


if __name__ == "__main__":
    main()