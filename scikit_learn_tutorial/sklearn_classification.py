"""
Scikit-learn 分类算法

本模块介绍scikit-learn中的分类算法，包括逻辑回归、决策树、随机森林、支持向量机(SVM)和K近邻(KNN)等。
分类是监督学习的一种，目标是预测离散的类别标签。
"""

import os
# 导入matplotlib配置（必须在任何其他matplotlib导入之前）
from matplotlib_config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from scikit_learn_tutorial.config import set_matplotlib_chinese


def load_data():
    """
    加载和准备数据集
    """
    # 加载乳腺癌数据集
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names


def evaluate_model(model, X_test, y_test, target_names, model_name):
    """
    评估分类模型并打印结果
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 如果模型支持predict_proba，则获取概率
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test) if hasattr(model, "decision_function") else None
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 打印评估结果
    print(f"\n{model_name} 评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} - 混淆矩阵')
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # 如果有概率预测，绘制ROC曲线
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率 (False Positive Rate)')
        plt.ylabel('真正例率 (True Positive Rate)')
        plt.title(f'{model_name} - ROC曲线')
        plt.legend(loc="lower right")
        # 确保目录存在
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/{model_name.lower().replace(' ', '_')}_roc_curve.png")
        plt.close()

        # 绘制精确率-召回率曲线
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2,
                 label=f'精确率-召回率曲线 (AP = {average_precision:.2f})')
        plt.axhline(y=sum(y_test) / len(y_test), color='red', linestyle='--',
                    label=f'基准 (正例比例 = {sum(y_test) / len(y_test):.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title(f'{model_name} - 精确率-召回率曲线')
        plt.legend(loc="lower left")
        # 确保目录存在
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/{model_name.lower().replace(' ', '_')}_precision_recall_curve.png")
        plt.close()

    return accuracy, precision, recall, f1


def logistic_regression_example():
    """
    逻辑回归分类示例
    """
    print("=" * 50)
    print("逻辑回归分类".center(50))
    print("=" * 50)

    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()

    print("\n逻辑回归是一种线性分类算法，适用于二分类问题。")
    print("它通过Sigmoid函数将线性模型的输出转换为概率。")

    # 创建并训练逻辑回归模型
    print("\n训练逻辑回归模型...")

    # 使用不同的正则化参数
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_accuracy = 0
    best_model = None
    best_C = None

    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # 在测试集上评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"C={C}: 准确率={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_C = C

    print(f"\n最佳正则化参数 C={best_C}, 准确率={best_accuracy:.4f}")

    # 查看模型系数
    print("\n特征重要性 (系数绝对值):")
    coef_importance = pd.DataFrame({
        '特征': feature_names,
        '系数': best_model.coef_[0],
        '绝对值': np.abs(best_model.coef_[0])
    }).sort_values('绝对值', ascending=False)

    print(coef_importance.head(10))

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = coef_importance.head(15)
    colors = ['red' if c < 0 else 'blue' for c in top_features['系数']]
    plt.barh(top_features['特征'], top_features['绝对值'], color=colors)
    plt.xlabel('系数绝对值')
    plt.title('逻辑回归 - 前15个重要特征')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/logistic_regression_feature_importance.png")
    plt.close()

    print("\n特征重要性图已保存为 'scikit_learn_tutorial/logistic_regression_feature_importance.png'")

    # 评估最佳模型
    evaluate_model(best_model, X_test, y_test, target_names, "逻辑回归")

    # 逻辑回归的优缺点
    print("\n逻辑回归的优点:")
    print("- 简单、高效、易于理解和实现")
    print("- 不容易过拟合（特别是使用正则化时）")
    print("- 提供概率输出")
    print("- 可以通过系数解释特征重要性")

    print("\n逻辑回归的缺点:")
    print("- 只能学习线性决策边界")
    print("- 对异常值敏感")
    print("- 假设特征之间相互独立")
    print("- 可能需要特征工程来捕捉非线性关系")


def decision_tree_example():
    """
    决策树分类示例
    """
    print("\n" + "=" * 50)
    print("决策树分类".center(50))
    print("=" * 50)

    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()

    print("\n决策树是一种非参数监督学习算法，通过一系列问题将数据划分为不同的类别。")
    print("它可以处理非线性关系，并自动进行特征选择。")

    # 创建并训练决策树模型
    print("\n训练决策树模型...")

    # 使用不同的最大深度
    max_depths = [3, 5, 7, 10, None]
    best_accuracy = 0
    best_model = None
    best_depth = None

    for depth in max_depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        # 在测试集上评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        depth_str = str(depth) if depth is not None else "None"
        print(f"max_depth={depth_str}: 准确率={accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_depth = depth

    depth_str = str(best_depth) if best_depth is not None else "None"
    print(f"\n最佳最大深度 max_depth={depth_str}, 准确率={best_accuracy:.4f}")

    # 查看特征重要性
    print("\n特征重要性:")
    feature_importance = pd.DataFrame({
        '特征': feature_names,
        '重要性': best_model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print(feature_importance.head(10))

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['特征'], top_features['重要性'])
    plt.xlabel('重要性')
    plt.title('决策树 - 前15个重要特征')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/decision_tree_feature_importance.png")
    plt.close()

    print("\n特征重要性图已保存为 'decision_tree_feature_importance.png'")

    # 可视化决策树（限制深度为3以便于可视化）
    plt.figure(figsize=(20, 10))
    simple_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    simple_tree.fit(X_train, y_train)
    plot_tree(simple_tree, feature_names=feature_names, class_names=target_names,
              filled=True, rounded=True, fontsize=10)
    plt.title('决策树可视化 (max_depth=3)')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/decision_tree_visualization.png")
    plt.close()

    print("\n决策树可视化已保存为 'decision_tree_visualization.png'")

    # 评估最佳模型
    evaluate_model(best_model, X_test, y_test, target_names, "决策树")

    # 决策树的优缺点
    print("\n决策树的优点:")
    print("- 易于理解和解释")
    print("- 可以处理数值型和分类型特征")
    print("- 可以处理多分类问题")
    print("- 不需要特征缩放")
    print("- 自动进行特征选择")

    print("\n决策树的缺点:")
    print("- 容易过拟合（特别是树很深时）")
    print("- 可能不如其他算法稳定（小的数据变化可能导致很大的树结构变化）")
    print("- 可能产生有偏的树（如果某些类别占主导）")
    print("- 对于某些关系难以学习（如XOR关系）")


def random_forest_example():
    """
    随机森林分类示例
    """
    print("\n" + "=" * 50)
    print("随机森林分类".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    print("\n随机森林是一种集成学习方法，通过组合多个决策树的预测来提高性能。")
    print("每棵树使用随机选择的特征子集和样本子集进行训练。")
    
    # 创建并训练随机森林模型
    print("\n训练随机森林模型...")
    
    # 使用不同的树数量
    n_estimators_list = [10, 50, 100, 200]
    best_accuracy = 0
    best_model = None
    best_n_estimators = None
    
    for n_estimators in n_estimators_list:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )
        model.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"n_estimators={n_estimators}: 准确率={accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_n_estimators = n_estimators
    
    print(f"\n最佳树数量 n_estimators={best_n_estimators}, 准确率={best_accuracy:.4f}")
    
    # 查看特征重要性
    print("\n特征重要性:")
    feature_importance = pd.DataFrame({
        '特征': feature_names,
        '重要性': best_model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print(feature_importance.head(10))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['特征'], top_features['重要性'])
    plt.xlabel('重要性')
    plt.title('随机森林 - 前15个重要特征')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/random_forest_feature_importance.png")
    plt.close()
    
    print("\n特征重要性图已保存为 'random_forest_feature_importance.png'")
    
    # 评估最佳模型
    evaluate_model(best_model, X_test, y_test, target_names, "随机森林")
    
    # 随机森林的优缺点
    print("\n随机森林的优点:")
    print("- 通常比单个决策树表现更好")
    print("- 不容易过拟合")
    print("- 可以处理高维数据")
    print("- 提供特征重要性评估")
    print("- 可以并行训练")
    
    print("\n随机森林的缺点:")
    print("- 比单个决策树更难解释")
    print("- 训练时间较长")
    print("- 预测时间较长")
    print("- 需要更多的内存")


def svm_example():
    """
    支持向量机(SVM)分类示例
    """
    print("\n" + "=" * 50)
    print("支持向量机分类".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    print("\n支持向量机(SVM)是一种强大的分类算法，通过寻找最大间隔超平面来分离不同类别。")
    print("通过核技巧，SVM可以处理非线性分类问题。")
    
    # 创建并训练SVM模型
    print("\n训练SVM模型...")
    
    # 使用不同的核函数和参数
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
    ]
    
    # 使用网格搜索找到最佳参数
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print("\n网格搜索结果:")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    # 使用最佳参数的模型
    best_model = grid_search.best_estimator_
    
    # 评估最佳模型
    evaluate_model(best_model, X_test, y_test, target_names, "支持向量机")
    
    # SVM的优缺点
    print("\nSVM的优点:")
    print("- 在高维空间中有效")
    print("- 对于非线性分类问题效果好")
    print("- 不容易过拟合（有正则化参数）")
    print("- 决策边界清晰")
    
    print("\nSVM的缺点:")
    print("- 对大规模数据集计算成本高")
    print("- 对特征缩放敏感")
    print("- 参数调优可能比较困难")
    print("- 不直接提供概率估计")


def knn_example():
    """
    K近邻(KNN)分类示例
    """
    print("\n" + "=" * 50)
    print("K近邻分类".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    print("\nK近邻(KNN)是一种简单但有效的分类算法，通过投票机制预测新样本的类别。")
    print("它基于'物以类聚'的原则，认为相似的样本应该属于同一类别。")
    
    # 创建并训练KNN模型
    print("\n训练KNN模型...")
    
    # 使用不同的邻居数量
    n_neighbors_list = [3, 5, 7, 9, 11]
    best_accuracy = 0
    best_model = None
    best_n_neighbors = None
    
    for n_neighbors in n_neighbors_list:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        
        # 在测试集上评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"n_neighbors={n_neighbors}: 准确率={accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_n_neighbors = n_neighbors
    
    print(f"\n最佳邻居数量 n_neighbors={best_n_neighbors}, 准确率={best_accuracy:.4f}")
    
    # 评估最佳模型
    evaluate_model(best_model, X_test, y_test, target_names, "K近邻")
    
    # KNN的优缺点
    print("\nKNN的优点:")
    print("- 简单易懂")
    print("- 不需要训练过程")
    print("- 可以处理多分类问题")
    print("- 对异常值不敏感")
    
    print("\nKNN的缺点:")
    print("- 计算成本高（需要计算所有训练样本的距离）")
    print("- 需要大量内存来存储训练数据")
    print("- 对特征缩放敏感")
    print("- 维度灾难（在高维空间中效果可能不好）")


def main():
    """主函数，运行所有示例"""
    # 设置matplotlib显示中文
    set_matplotlib_chinese()
    logistic_regression_example()
    decision_tree_example()
    random_forest_example()
    svm_example()
    knn_example()


if __name__ == "__main__":
    main()