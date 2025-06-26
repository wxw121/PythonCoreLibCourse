"""
Scikit-learn 实际应用案例

本模块展示如何将scikit-learn应用于实际问题，包括：
1. 文本分类 - 使用新闻数据集
2. 图像识别 - 使用手写数字数据集
3. 推荐系统 - 使用电影评分数据
"""

import os
# 导入matplotlib配置（必须在任何其他matplotlib导入之前）
from matplotlib_config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups, load_digits
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error
)

from scikit_learn_tutorial.config import set_matplotlib_chinese


def text_classification_example():
    """
    使用20 Newsgroups数据集进行文本分类
    展示如何处理文本数据并构建分类器
    """
    print("=" * 50)
    print("文本分类示例".center(50))
    print("=" * 50)
    
    print("\n本示例使用20 Newsgroups数据集进行文本分类。")
    print("我们将尝试区分'comp.graphics'和'sci.space'两个类别的文章。")
    
    # 加载数据
    categories = ['comp.graphics', 'sci.space']
    print("\n1. 加载数据...")
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')  # 移除干扰信息
    )
    
    print(f"数据集大小: {len(newsgroups.data)} 文档")
    print(f"类别: {newsgroups.target_names}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        newsgroups.data, newsgroups.target,
        test_size=0.3, random_state=42
    )
    
    # 创建文本分类流水线
    print("\n2. 创建文本分类流水线...")
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )),
        ('clf', MultinomialNB())
    ])

    # 训练模型
    print("3. 训练模型...")
    text_clf.fit(X_train, y_train)

    # 评估模型
    print("\n4. 评估模型...")
    y_pred = text_clf.predict(X_test)

    print("\n分类报告:")
    print(classification_report(
        y_test, y_pred,
        target_names=categories
    ))

    # 分析错误预测
    print("\n5. 错误分析...")
    errors = []
    for doc, true_label, pred_label in zip(X_test, y_test, y_pred):
        if true_label != pred_label:
            errors.append({
                'text': doc[:200] + '...',  # 只显示前200个字符
                'true': categories[true_label],
                'pred': categories[pred_label]
            })

    if errors:
        print("\n错误预测示例:")
        for i, error in enumerate(errors[:3], 1):
            print(f"\n错误 {i}:")
            print(f"文本: {error['text']}")
            print(f"真实类别: {error['true']}")
            print(f"预测类别: {error['pred']}")

    # 特征重要性分析
    print("\n6. 特征重要性分析...")
    feature_names = text_clf.named_steps['tfidf'].get_feature_names_out()
    feature_importances = np.abs(text_clf.named_steps['clf'].coef_[0])

    # 获取最重要的特征
    top_features = pd.DataFrame({
        '特征': feature_names,
        '重要性': feature_importances
    }).sort_values('重要性', ascending=False)

    print("\n最重要的10个特征:")
    print(top_features.head(10))

    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    plt.bar(range(20), feature_importances[np.argsort(feature_importances)[-20:]])
    plt.xticks(
        range(20),
        feature_names[np.argsort(feature_importances)[-20:]],
        rotation=45,
        ha='right'
    )
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/text_classification_features.png")
    plt.close()

    print("\n特征重要性图已保存为 'images/text_classification_features.png'")


def image_recognition_example():
    """
    使用手写数字数据集进行图像识别
    展示如何处理图像数据并构建分类器
    """
    print("\n" + "=" * 50)
    print("图像识别示例".center(50))
    print("=" * 50)

    print("\n本示例使用手写数字数据集进行图像识别。")

    # 加载数据
    print("\n1. 加载数据...")
    digits = load_digits()
    print(f"图像大小: {digits.images[0].shape}")
    print(f"数据集大小: {len(digits.images)} 图像")

    # 数据预处理
    print("\n2. 数据预处理...")
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 创建并训练模型
    print("\n3. 训练模型...")
    image_clf = SVC(kernel='rbf', random_state=42)
    image_clf.fit(X_train, y_train)

    # 评估模型
    print("\n4. 评估模型...")
    y_pred = image_clf.predict(X_test)

    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 可视化结果
    print("\n5. 可视化结果...")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/image_recognition_confusion_matrix.png")
    plt.close()

    print("\n混淆矩阵已保存为 'images/image_recognition_confusion_matrix.png'")

    # 显示一些错误预测的例子
    print("\n6. 错误预测分析...")
    errors = np.where(y_pred != y_test)[0]
    if len(errors) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < min(len(errors), 10):
                idx = errors[i]
                ax.imshow(
                    digits.images[y_test.index[idx]],
                    cmap=plt.cm.gray_r
                )
                ax.set_title(f'真实:{y_test[idx]}\n预测:{y_pred[idx]}')
            ax.axis('off')

        plt.tight_layout()
        # 确保目录存在
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/image_recognition_errors.png")
        plt.close()

        print("\n错误预测示例已保存为 'images/image_recognition_errors.png'")


def recommendation_system_example():
    """
    使用矩阵分解构建简单的推荐系统
    展示如何处理评分数据并生成推荐
    """
    print("\n" + "=" * 50)
    print("推荐系统示例".center(50))
    print("=" * 50)

    print("\n本示例创建一个简单的基于矩阵分解的推荐系统。")

    # 创建示例数据
    print("\n1. 创建示例数据...")
    np.random.seed(42)
    n_users = 100
    n_items = 50
    n_factors = 10

    # 生成真实的潜在因子
    user_factors = np.random.normal(0, 1, (n_users, n_factors))
    item_factors = np.random.normal(0, 1, (n_items, n_factors))

    # 生成真实的评分矩阵
    true_ratings = np.dot(user_factors, item_factors.T)
    true_ratings = (true_ratings - true_ratings.min()) / (true_ratings.max() - true_ratings.min()) * 4 + 1

    # 添加一些噪声
    noise = np.random.normal(0, 0.1, true_ratings.shape)
    observed_ratings = true_ratings + noise

    # 随机遮盖一些评分（创建稀疏矩阵）
    mask = np.random.random(observed_ratings.shape) > 0.8
    sparse_ratings = np.where(mask, observed_ratings, 0)

    print(f"用户数量: {n_users}")
    print(f"物品数量: {n_items}")
    print(f"评分密度: {np.mean(mask):.2%}")

    # 使用NMF进行矩阵分解
    print("\n2. 训练推荐模型...")
    nmf = NMF(
        n_components=n_factors,
        random_state=42,
        max_iter=200
    )

    # 获取用户和物品的潜在特征
    user_features = nmf.fit_transform(sparse_ratings)
    item_features = nmf.components_

    # 生成预测评分
    predicted_ratings = np.dot(user_features, item_features)

    # 评估模型
    print("\n3. 评估模型...")
    # 只在有观察值的位置计算误差
    mask = sparse_ratings != 0
    rmse = np.sqrt(mean_squared_error(
        observed_ratings[mask],
        predicted_ratings[mask]
    ))
    print(f"RMSE: {rmse:.4f}")

    # 为一个示例用户生成推荐
    print("\n4. 生成推荐示例...")
    user_id = 0  # 选择第一个用户作为示例

    # 获取用户未评分的物品
    unrated_items = np.where(sparse_ratings[user_id] == 0)[0]

    # 预测这些物品的评分
    predicted_ratings_user = predicted_ratings[user_id]

    # 获取top-5推荐
    top_items = np.argsort(predicted_ratings_user[unrated_items])[-5:][::-1]

    print(f"\n用户{user_id}的Top-5推荐:")
    for i, item_id in enumerate(top_items, 1):
        print(f"推荐 {i}: 物品{item_id} (预测评分: {predicted_ratings_user[item_id]:.2f})")

    # 可视化评分分布
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(observed_ratings[mask], bins=20, alpha=0.5, label='观察到的评分')
    plt.hist(predicted_ratings[mask], bins=20, alpha=0.5, label='预测的评分')
    plt.title('评分分布对比')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(observed_ratings[mask], predicted_ratings[mask], alpha=0.1)
    plt.plot([1, 5], [1, 5], 'r--')
    plt.xlabel('观察到的评分')
    plt.ylabel('预测的评分')
    plt.title('预测 vs 观察')

    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/recommendation_system_analysis.png")
    plt.close()

    print("\n分析图表已保存为 'recommendation_system_analysis.png'")


def main():
    """主函数，运行所有示例"""
    # 设置matplotlib显示中文
    set_matplotlib_chinese()
    text_classification_example()
    image_recognition_example()
    recommendation_system_example()


if __name__ == "__main__":
    main()
