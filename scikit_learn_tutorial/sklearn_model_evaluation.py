"""
Scikit-learn 模型评估

本模块介绍scikit-learn中的模型评估技术，包括交叉验证、网格搜索、评估指标和学习曲线等。
模型评估是机器学习流程中的关键步骤，帮助我们理解模型的性能和优化方向。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, 
    learning_curve, validation_curve, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, mean_squared_error, r2_score, average_precision_score
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
    
    return X_train, X_test, y_train, y_test, feature_names, target_names


def cross_validation_example():
    """
    交叉验证示例
    """
    print("=" * 50)
    print("交叉验证".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    print("\n交叉验证是一种评估模型性能的方法，它通过将数据多次划分为训练集和验证集来获得更可靠的性能估计。")
    
    # 创建模型和预处理流水线
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', random_state=42))
    ])
    
    # 1. 基本交叉验证
    print("\n1. 基本交叉验证 (5折)")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
    print("交叉验证分数:")
    print(f"各折分数: {cv_scores}")
    print(f"平均分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 2. 使用不同的评估指标
    print("\n2. 使用不同的评估指标")
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=metric)
        print(f"{metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # 3. 自定义交叉验证分割
    print("\n3. 自定义交叉验证分割")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # 获取当前折的训练和验证数据
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        # 训练和评估模型
        pipeline.fit(X_fold_train, y_fold_train)
        score = pipeline.score(X_fold_val, y_fold_val)
        fold_scores.append(score)
        
        print(f"第{fold+1}折得分: {score:.4f}")
    
    print(f"\n平均得分: {np.mean(fold_scores):.4f}")
    
    # 可视化交叉验证结果
    plt.figure(figsize=(10, 6))
    plt.boxplot(cv_scores)
    plt.title('交叉验证分数分布')
    plt.ylabel('准确率')
    plt.xticks([1], ['SVM'])
    plt.grid(True)
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/cross_validation_scores.png")
    plt.close()
    
    print("\n交叉验证分数分布图已保存为 'cross_validation_scores.png'")


def learning_curves_example():
    """
    学习曲线示例
    """
    print("\n" + "=" * 50)
    print("学习曲线".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    print("\n学习曲线显示了模型的性能如何随训练数据量的增加而变化。")
    print("它可以帮助诊断模型是否存在过拟合或欠拟合问题。")
    
    # 创建模型和预处理流水线
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', random_state=42))
    ])
    
    # 计算学习曲线
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        n_jobs=-1
    )
    
    # 计算平均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # 可视化学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='训练集得分', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    
    plt.plot(train_sizes, val_mean, label='验证集得分', color='green', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color='green')
    
    plt.xlabel('训练样本数')
    plt.ylabel('准确率')
    plt.title('学习曲线')
    plt.legend(loc='lower right')
    plt.grid(True)
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/learning_curves.png")
    plt.close()
    
    print("\n学习曲线已保存为 'learning_curves.png'")
    
    # 分析学习曲线
    print("\n学习曲线分析:")
    print(f"最终训练集得分: {train_mean[-1]:.4f} (+/- {train_std[-1] * 2:.4f})")
    print(f"最终验证集得分: {val_mean[-1]:.4f} (+/- {val_std[-1] * 2:.4f})")
    
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        print("\n诊断: 可能存在过拟合")
        print("建议:")
        print("- 增加正则化")
        print("- 减少模型复杂度")
        print("- 收集更多数据")
    elif train_mean[-1] < 0.8:
        print("\n诊断: 可能存在欠拟合")
        print("建议:")
        print("- 增加模型复杂度")
        print("- 添加更多特征")
        print("- 减少正则化")
    else:
        print("\n诊断: 模型表现良好")
        print("- 训练集和验证集的得分都较高")
        print("- 两者之间的差距较小")


def validation_curves_example():
    """
    验证曲线示例
    """
    print("\n" + "=" * 50)
    print("验证曲线".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    print("\n验证曲线显示了模型的性能如何随超参数的变化而变化。")
    print("它可以帮助我们选择最佳的超参数值。")
    
    # 创建模型和预处理流水线
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', random_state=42))
    ])
    
    # 计算验证曲线
    param_range = np.logspace(-6, 3, 10)
    train_scores, val_scores = validation_curve(
        pipeline, X_train, y_train,
        param_name="classifier__C",
        param_range=param_range,
        cv=5,
        n_jobs=-1
    )
    
    # 计算平均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # 可视化验证曲线
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, train_mean, label='训练集得分', color='blue', marker='o')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color='blue')
    
    plt.semilogx(param_range, val_mean, label='验证集得分', color='green', marker='o')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color='green')
    
    plt.xlabel('SVM正则化参数 (C)')
    plt.ylabel('准确率')
    plt.title('验证曲线')
    plt.legend(loc='lower right')
    plt.grid(True)
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/validation_curves.png")
    plt.close()
    
    print("\n验证曲线已保存为 'validation_curves.png'")
    
    # 找出最佳参数
    best_idx = np.argmax(val_mean)
    best_C = param_range[best_idx]
    best_score = val_mean[best_idx]
    
    print(f"\n最佳参数 C: {best_C:.6f}")
    print(f"最佳验证集得分: {best_score:.4f}")


def grid_search_example():
    """
    网格搜索示例
    """
    print("\n" + "=" * 50)
    print("网格搜索".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    print("\n网格搜索是一种自动化的方法，用于系统地搜索最佳超参数组合。")
    
    # 创建模型和预处理流水线
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # 定义参数网格
    param_grid = {
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # 创建网格搜索对象
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("\n开始网格搜索...")
    grid_search.fit(X_train, y_train)
    
    print("\n网格搜索结果:")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
    
    # 在测试集上评估最佳模型
    y_pred = grid_search.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    print(f"测试集得分: {test_score:.4f}")
    
    # 可视化参数重要性
    results = pd.DataFrame(grid_search.cv_results_)

    # 创建参数得分热图
    pivot_tables = {}
    for param1 in ['param_classifier__n_estimators', 'param_classifier__max_depth']:
        for param2 in ['param_classifier__min_samples_split']:
            if param1 != param2:
                pivot = pd.pivot_table(
                    results,
                    values='mean_test_score',
                    index=param1,
                    columns=param2
                )
                pivot_tables[(param1, param2)] = pivot
    
    # 绘制热图
    n_plots = len(pivot_tables)
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, ((param1, param2), pivot) in enumerate(pivot_tables.items()):
        sns.heatmap(pivot, annot=True, cmap='viridis', ax=axes[i])
        axes[i].set_title(f'{param1} vs {param2}')
    
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/grid_search_heatmap.png")
    plt.close()
    
    print("\n参数重要性热图已保存为 'grid_search_heatmap.png'")
    
    # 可视化最佳模型的特征重要性
    best_model = grid_search.best_estimator_
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': feature_importances
    }).sort_values('重要性', ascending=False)
    
    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='重要性', y='特征', data=importance_df.head(15))
    plt.title('随机森林 - 前15个重要特征')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/grid_search_feature_importance.png")
    plt.close()
    
    print("\n特征重要性图已保存为 'grid_search_feature_importance.png'")


def classification_metrics_example():
    """
    分类评估指标示例
    """
    print("\n" + "=" * 50)
    print("分类评估指标".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    
    print("\n分类评估指标用于衡量分类模型的性能。")
    print("不同的指标适用于不同的问题和场景。")
    
    # 创建并训练模型
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    model.fit(X_train, y_train)
    
    # 获取预测和概率
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. 基本分类指标
    print("\n1. 基本分类指标:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 2. 混淆矩阵
    print("\n2. 混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/confusion_matrix_metrics.png")
    plt.close()
    
    print("\n混淆矩阵已保存为 'confusion_matrix_metrics.png'")
    
    # 3. 分类报告
    print("\n3. 分类报告:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # 4. ROC曲线和AUC
    print("\n4. ROC曲线和AUC:")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)')
    plt.ylabel('真正例率 (True Positive Rate)')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/roc_curve_metrics.png")
    plt.close()
    
    print("\nROC曲线已保存为 'roc_curve_metrics.png'")
    
    # 5. 精确率-召回率曲线
    print("\n5. 精确率-召回率曲线:")
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    average_precision = average_precision_score(y_test, y_prob)
    print(f"平均精确率: {average_precision:.4f}")
    
    # 绘制精确率-召回率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'精确率-召回率曲线 (AP = {average_precision:.2f})')
    plt.axhline(y=sum(y_test) / len(y_test), color='red', linestyle='--',
                label=f'基准 (正例比例 = {sum(y_test) / len(y_test):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/precision_recall_curve_metrics.png")
    plt.close()
    
    print("\n精确率-召回率曲线已保存为 'precision_recall_curve_metrics.png'")
    
    # 6. 不同阈值下的指标
    print("\n6. 不同阈值下的指标:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n阈值\t准确率\t精确率\t召回率\tF1分数")
    print("-" * 50)
    
    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)
        
        acc = accuracy_score(y_test, y_pred_threshold)
        prec = precision_score(y_test, y_pred_threshold)
        rec = recall_score(y_test, y_pred_threshold)
        f1 = f1_score(y_test, y_pred_threshold)
        
        print(f"{threshold:.1f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}")


def regression_metrics_example():
    """
    回归评估指标示例
    """
    print("\n" + "=" * 50)
    print("回归评估指标".center(50))
    print("=" * 50)
    
    # 加载波士顿房价数据集（如果可用）
    try:
        boston = datasets.load_boston()
        X = boston.data
        y = boston.target
        dataset_name = "Boston Housing"
    except:
        # 如果波士顿数据集不可用，使用加州房价数据集
        california = datasets.fetch_california_housing()
        X = california.data
        y = california.target
        dataset_name = "California Housing"
    
    print(f"\n使用{dataset_name}数据集")
    print("\n回归评估指标用于衡量回归模型的性能。")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建并训练模型
    from sklearn.ensemble import RandomForestRegressor
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    model.fit(X_train, y_train)
    
    # 获取预测
    y_pred = model.predict(X_test)
    
    # 1. 基本回归指标
    print("\n1. 基本回归指标:")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"R²分数: {r2:.4f}")
    
    # 2. 可视化预测值与真实值的对比
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 真实值')
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/regression_predictions.png")
    plt.close()
    
    print("\n预测值与真实值的对比图已保存为 'regression_predictions.png'")
    
    # 3. 残差图
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/regression_residuals.png")
    plt.close()
    
    print("\n残差图已保存为 'regression_residuals.png'")
    
    # 4. 残差分布
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('残差')
    plt.ylabel('频率')
    plt.title('残差分布')
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/regression_residuals_distribution.png")
    plt.close()
    
    print("\n残差分布图已保存为 'regression_residuals_distribution.png'")


def main():
    """主函数，运行所有示例"""
    # 设置matplotlib显示中文
    set_matplotlib_chinese()
    cross_validation_example()
    learning_curves_example()
    validation_curves_example()
    grid_search_example()
    classification_metrics_example()
    regression_metrics_example()


if __name__ == "__main__":
    main()