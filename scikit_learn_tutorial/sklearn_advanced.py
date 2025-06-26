"""
Scikit-learn 高级主题

本模块涵盖scikit-learn的高级主题，包括：
1. 模型集成 - 投票分类器、Bagging和Boosting
2. 特征选择 - 过滤法、包装法和嵌入法
3. 异常检测 - Isolation Forest和One-Class SVM
4. 管道和参数优化 - 复杂管道和自动化调优
"""

import os
# 导入matplotlib配置（必须在任何其他matplotlib导入之前）
from matplotlib_config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel
)
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier,
    GradientBoostingClassifier, IsolationForest
)
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve, auc,
    precision_recall_curve
)

from scikit_learn_tutorial.config import set_matplotlib_chinese


def ensemble_learning_example():
    """
    模型集成学习示例
    展示不同的集成方法：投票、Bagging和Boosting
    """
    print("=" * 50)
    print("模型集成学习示例".center(50))
    print("=" * 50)

    # 生成数据
    print("\n1. 生成示例数据...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 1. 投票分类器
    print("\n2. 投票分类器示例...")

    # 创建基分类器
    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    clf3 = SVC(probability=True, random_state=42)

    # 创建投票分类器
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', clf1),
            ('rf', clf2),
            ('svc', clf3)
        ],
        voting='soft'  # 使用概率进行投票
    )

    # 评估各个分类器
    clfs = [clf1, clf2, clf3, voting_clf]
    clf_names = ['逻辑回归', '随机森林', 'SVM', '投票分类器']

    for clf, name in zip(clfs, clf_names):
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        print(f"{name} 交叉验证得分: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # 2. Bagging示例
    print("\n3. Bagging示例...")

    # 创建基于决策树的Bagging分类器
    bagging = BaggingClassifier(
        DecisionTreeClassifier(),
        n_estimators=100,
        max_samples=0.7,
        max_features=0.7,
        random_state=42
    )

    bagging.fit(X_train, y_train)
    bagging_pred = bagging.predict(X_test)

    print("\nBagging分类器性能:")
    print(classification_report(y_test, bagging_pred))

    # 3. Boosting示例
    print("\n4. Boosting示例...")

    # 创建梯度提升分类器
    boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    boosting.fit(X_train, y_train)
    boosting_pred = boosting.predict(X_test)

    print("\nGradient Boosting分类器性能:")
    print(classification_report(y_test, boosting_pred))

    # 训练所有分类器
    voting_clf.fit(X_train, y_train)
    
    # 比较ROC曲线
    plt.figure(figsize=(10, 6))

    classifiers = [
        (voting_clf, '投票分类器'),
        (bagging, 'Bagging'),
        (boosting, 'Gradient Boosting')
    ]

    for clf, name in classifiers:
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title('不同集成方法的ROC曲线比较')
    plt.legend(loc="lower right")
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/ensemble_roc_curves.png")
    plt.close()

    print("\nROC曲线比较已保存为 'ensemble_roc_curves.png'")


def feature_selection_example():
    """
    特征选择示例
    展示不同的特征选择方法：过滤法、包装法和嵌入法
    """
    print("\n" + "=" * 50)
    print("特征选择示例".center(50))
    print("=" * 50)

    # 生成数据
    print("\n1. 生成示例数据...")
    X, y = make_classification(
        n_samples=1000,
        n_features=50,  # 50个特征
        n_informative=10,  # 只有10个特征是有信息量的
        n_redundant=25,
        n_repeated=5,
        n_classes=2,
        random_state=42
    )

    feature_names = [f'特征_{i}' for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 1. 过滤法 - 单变量特征选择
    print("\n2. 过滤法示例...")

    # 使用F-检验选择特征
    selector = SelectKBest(score_func=f_classif, k=10)
    X_filtered = selector.fit_transform(X_train, y_train)

    # 获取所选特征的得分
    scores = pd.DataFrame({
        '特征': feature_names,
        'F值': selector.scores_,
        'P值': selector.pvalues_
    }).sort_values('F值', ascending=False)

    print("\n按F检验得分排序的前10个特征:")
    print(scores.head(10))

    # 2. 包装法 - 递归特征消除
    print("\n3. 包装法示例...")

    # 使用随机森林作为基分类器进行RFE
    rfe = RFE(
        estimator=RandomForestClassifier(random_state=42),
        n_features_to_select=10,
        step=1
    )
    X_wrapped = rfe.fit_transform(X_train, y_train)

    # 获取特征排名
    feature_ranking = pd.DataFrame({
        '特征': feature_names,
        '是否选择': rfe.support_,
        '排名': rfe.ranking_
    }).sort_values('排名')

    print("\n递归特征消除结果:")
    print(feature_ranking[feature_ranking['是否选择']].head(10))

    # 3. 嵌入法 - 基于模型的特征选择
    print("\n4. 嵌入法示例...")

    # 使用带L1正则化的逻辑回归
    from sklearn.linear_model import LogisticRegression

    selector = SelectFromModel(
        LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=42),
        prefit=False
    )
    X_embedded = selector.fit_transform(X_train, y_train)

    # 获取特征重要性
    importance = pd.DataFrame({
        '特征': feature_names,
        '重要性': np.abs(selector.estimator_.coef_[0])
    }).sort_values('重要性', ascending=False)

    print("\n基于L1正则化的特征重要性:")
    print(importance.head(10))

    # 比较不同特征选择方法的性能
    print("\n5. 比较不同特征选择方法的性能...")

    # 创建基准分类器
    base_clf = RandomForestClassifier(random_state=42)

    # 创建不同的特征选择流水线
    pipelines = {
        '无特征选择': Pipeline([
            ('classifier', base_clf)
        ]),
        '过滤法': Pipeline([
            ('selector', SelectKBest(score_func=f_classif, k=10)),
            ('classifier', base_clf)
        ]),
        '包装法': Pipeline([
            ('selector', RFE(estimator=base_clf, n_features_to_select=10)),
            ('classifier', base_clf)
        ]),
        '嵌入法': Pipeline([
            ('selector', SelectFromModel(LogisticRegression(
                C=1, penalty='l1', solver='liblinear', random_state=42
            ))),
            ('classifier', base_clf)
        ])
    }

    # 评估每种方法
    results = []
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        results.append({
            '方法': name,
            '平均得分': scores.mean(),
            '标准差': scores.std()
        })

    results_df = pd.DataFrame(results)
    print("\n不同特征选择方法的性能比较:")
    print(results_df)

    # 可视化比较结果
    plt.figure(figsize=(10, 6))
    plt.bar(
        results_df['方法'],
        results_df['平均得分'],
        yerr=results_df['标准差'],
        capsize=5
    )
    plt.title('不同特征选择方法的性能比较')
    plt.ylabel('交叉验证得分')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/feature_selection_comparison.png")
    plt.close()

    print("\n特征选择方法比较图已保存为 'feature_selection_comparison.png'")


def anomaly_detection_example():
    """
    异常检测示例
    展示不同的异常检测方法：Isolation Forest和One-Class SVM
    """
    print("\n" + "=" * 50)
    print("异常检测示例".center(50))
    print("=" * 50)

    # 生成数据
    print("\n1. 生成示例数据...")
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # 生成正常点
    X, _ = make_blobs(
        n_samples=n_inliers,
        n_features=2,
        centers=1,
        cluster_std=0.5,
        random_state=42
    )

    # 生成异常点
    X_outliers = np.random.uniform(
        low=-4,
        high=4,
        size=(n_outliers, 2)
    )

    # 合并正常点和异常点
    X = np.r_[X, X_outliers]

    # 创建真实标签（1表示正常，-1表示异常）
    y = np.ones(n_samples)
    y[-n_outliers:] = -1

    # 1. Isolation Forest
    print("\n2. Isolation Forest示例...")

    iso_forest = IsolationForest(
        contamination=outliers_fraction,
        random_state=42
    )
    y_pred_forest = iso_forest.fit_predict(X)

    # 2. One-Class SVM
    print("\n3. One-Class SVM示例...")

    one_class_svm = OneClassSVM(
        kernel='rbf',
        gamma='auto',
        nu=outliers_fraction
    )
    y_pred_svm = one_class_svm.fit_predict(X)
    
    # 评估结果
    print("\n4. 评估结果...")
    
    def evaluate_anomaly_detector(y_true, y_pred, method_name):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{method_name}性能指标:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        return accuracy, precision, recall, f1
    
    # 评估两种方法
    forest_metrics = evaluate_anomaly_detector(y, y_pred_forest, "Isolation Forest")
    svm_metrics = evaluate_anomaly_detector(y, y_pred_svm, "One-Class SVM")
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # Isolation Forest结果
    plt.subplot(121)
    plt.scatter(X[y_pred_forest == 1, 0], X[y_pred_forest == 1, 1],
                c='blue', label='正常点')
    plt.scatter(X[y_pred_forest == -1, 0], X[y_pred_forest == -1, 1],
                c='red', label='异常点')
    plt.title('Isolation Forest检测结果')
    plt.legend()
    
    # One-Class SVM结果
    plt.subplot(122)
    plt.scatter(X[y_pred_svm == 1, 0], X[y_pred_svm == 1, 1],
                c='blue', label='正常点')
    plt.scatter(X[y_pred_svm == -1, 0], X[y_pred_svm == -1, 1],
                c='red', label='异常点')
    plt.title('One-Class SVM检测结果')
    plt.legend()
    
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/anomaly_detection_results.png")
    plt.close()
    
    print("\n异常检测结果已保存为 'anomaly_detection_results.png'")
    
    # 比较两种方法的性能
    methods = ['Isolation Forest', 'One-Class SVM']
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    
    results = pd.DataFrame({
        'Isolation Forest': forest_metrics,
        'One-Class SVM': svm_metrics
    }, index=metrics)
    
    print("\n方法比较:")
    print(results)
    
    # 可视化性能比较
    plt.figure(figsize=(10, 6))
    results.plot(kind='bar', rot=0)
    plt.title('异常检测方法性能比较')
    plt.ylabel('分数')
    plt.legend(title='方法')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/anomaly_detection_comparison.png")
    plt.close()
    
    print("\n性能比较图已保存为 'anomaly_detection_comparison.png'")


def pipeline_optimization_example():
    """
    管道和参数优化示例
    展示如何构建复杂的管道并进行自动化参数调优
    """
    print("\n" + "=" * 50)
    print("管道和参数优化示例".center(50))
    print("=" * 50)
    
    # 生成数据
    print("\n1. 生成示例数据...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 1. 构建复杂管道
    print("\n2. 构建复杂管道...")
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import GridSearchCV
    
    # 创建一个包含预处理、特征选择和分类器的管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(include_bias=False)),
        ('selector', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # 2. 定义参数网格
    print("\n3. 定义参数网格...")
    
    param_grid = {
        'selector__k': [5, 10, 15],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    
    # 3. 使用网格搜索进行参数优化
    print("\n4. 执行网格搜索...")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # 4. 分析结果
    print("\n5. 分析优化结果...")
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
    
    # 在测试集上评估
    y_pred = grid_search.predict(X_test)
    print("\n测试集性能:")
    print(classification_report(y_test, y_pred))
    
    # 分析参数重要性
    results = pd.DataFrame(grid_search.cv_results_)
    
    # 创建参数组合的性能热图
    param_scores = results.pivot_table(
        values='mean_test_score',
        index=['param_selector__k', 'param_classifier__min_samples_split'],
        columns=['param_classifier__n_estimators', 'param_classifier__max_depth']
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(param_scores, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('参数组合的性能热图')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/parameter_optimization_heatmap.png")
    plt.close()
    
    print("\n参数优化热图已保存为 'parameter_optimization_heatmap.png'")
    
    # 分析特征重要性
    best_pipeline = grid_search.best_estimator_
    feature_importance = best_pipeline.named_steps['classifier'].feature_importances_
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': [f'特征_{i}' for i in range(len(feature_importance))],
        '重要性': feature_importance
    }).sort_values('重要性', ascending=False)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='重要性', y='特征', data=importance_df)
    plt.title('选中特征的重要性')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/feature_importance.png")
    plt.close()
    
    print("\n特征重要性图已保存为 'feature_importance.png'")


def main():
    """主函数，运行所有示例"""
    # 设置matplotlib显示中文
    set_matplotlib_chinese()
    ensemble_learning_example()
    feature_selection_example()
    anomaly_detection_example()
    pipeline_optimization_example()


if __name__ == "__main__":
    main()