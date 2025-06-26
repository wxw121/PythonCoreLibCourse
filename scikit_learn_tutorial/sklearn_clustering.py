"""
Scikit-learn 聚类算法

本模块介绍scikit-learn中的聚类算法，包括K-means聚类、层次聚类和DBSCAN等。
聚类是无监督学习的一种，目标是将相似的样本分组在一起。
"""

import os
# 导入matplotlib配置（必须在任何其他matplotlib导入之前）
from matplotlib_config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN
)
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from scipy.cluster.hierarchy import dendrogram

from scikit_learn_tutorial.config import set_matplotlib_chinese


def load_data():
    """
    加载和准备数据集
    """
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target  # 真实标签，仅用于评估
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names, target_names


def visualize_clusters(X, labels, centers=None, title="聚类结果"):
    """
    使用PCA将数据降至2维并可视化聚类结果
    """
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 创建DataFrame以便于可视化
    df = pd.DataFrame({
        'x': X_pca[:, 0],
        'y': X_pca[:, 1],
        'cluster': labels
    })
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    sns.scatterplot(
        x='x', y='y', 
        hue='cluster', 
        palette='viridis',
        data=df,
        legend='full',
        alpha=0.7
    )
    
    # 如果提供了中心点，则绘制中心点
    if centers is not None:
        centers_pca = pca.transform(centers)
        plt.scatter(
            centers_pca[:, 0], centers_pca[:, 1],
            s=200, marker='X', c='red', label='中心点'
        )
    
    plt.title(title)
    plt.xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2f})')
    plt.legend(title='聚类')
    
    return plt


def evaluate_clustering(X, labels, true_labels=None):
    """
    评估聚类结果
    """
    # 内部评估指标（不需要真实标签）
    silhouette = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    print("\n聚类评估指标:")
    print(f"轮廓系数 (Silhouette Coefficient): {silhouette:.4f} (越高越好，范围: [-1, 1])")
    print(f"Calinski-Harabasz指数: {calinski_harabasz:.4f} (越高越好)")
    print(f"Davies-Bouldin指数: {davies_bouldin:.4f} (越低越好)")
    
    # 如果提供了真实标签，计算外部评估指标
    if true_labels is not None:
        from sklearn.metrics import (
            adjusted_rand_score, normalized_mutual_info_score,
            adjusted_mutual_info_score
        )
        
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        ami = adjusted_mutual_info_score(true_labels, labels)
        
        print("\n与真实标签比较 (外部评估指标):")
        print(f"调整兰德指数 (ARI): {ari:.4f} (越高越好，范围: [-1, 1])")
        print(f"归一化互信息 (NMI): {nmi:.4f} (越高越好，范围: [0, 1])")
        print(f"调整互信息 (AMI): {ami:.4f} (越高越好，范围: [0, 1])")
        
        return silhouette, calinski_harabasz, davies_bouldin, ari, nmi, ami
    
    return silhouette, calinski_harabasz, davies_bouldin


def kmeans_example():
    """
    K-means聚类示例
    """
    print("=" * 50)
    print("K-means聚类".center(50))
    print("=" * 50)
    
    # 加载数据
    X, y, feature_names, target_names = load_data()
    
    print("\nK-means是最流行的聚类算法之一，它将数据点分配给K个聚类中心。")
    print("算法通过迭代最小化每个点到其分配的聚类中心的距离。")
    
    # 确定最佳聚类数
    print("\n确定最佳聚类数...")
    
    # 计算不同K值的肘部图
    inertia = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        print(f"K={k}: 惯性={kmeans.inertia_:.2f}, 轮廓系数={silhouette_scores[-1]:.4f}")
    
    # 可视化肘部图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('聚类数 (K)')
    plt.ylabel('惯性 (Inertia)')
    plt.title('肘部图')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'o-')
    plt.xlabel('聚类数 (K)')
    plt.ylabel('轮廓系数')
    plt.title('轮廓系数图')
    plt.grid(True)
    
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/kmeans_elbow_silhouette.png")
    plt.close()
    
    print("\n肘部图和轮廓系数图已保存为 'kmeans_elbow_silhouette.png'")
    
    # 使用K=3（鸢尾花数据集有3个类别）
    k = 3
    print(f"\n使用K={k}进行K-means聚类...")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # 评估聚类结果
    evaluate_clustering(X, labels, y)
    
    # 可视化聚类结果
    plt_kmeans = visualize_clusters(X, labels, centers, f"K-means聚类 (K={k})")
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt_kmeans.savefig("images/kmeans_clusters.png")
    plt_kmeans.close()
    
    print("\nK-means聚类结果已保存为 'kmeans_clusters.png'")
    
    # K-means的优缺点
    print("\nK-means的优点:")
    print("- 简单易懂")
    print("- 计算效率高，可扩展到大型数据集")
    print("- 当聚类是球形且大小相似时效果好")
    
    print("\nK-means的缺点:")
    print("- 需要预先指定聚类数K")
    print("- 对初始聚类中心敏感")
    print("- 只能发现球形聚类")
    print("- 对异常值敏感")
    print("- 结果可能不稳定（取决于初始化）")


def plot_dendrogram(model, **kwargs):
    """
    创建层次聚类的树状图
    """
    # 从模型中提取距离信息
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # 叶节点
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([
        model.children_, model.distances_, counts
    ]).astype(float)

    # 绘制树状图
    dendrogram(linkage_matrix, **kwargs)


def hierarchical_clustering_example():
    """
    层次聚类示例
    """
    print("\n" + "=" * 50)
    print("层次聚类".center(50))
    print("=" * 50)
    
    # 加载数据
    X, y, feature_names, target_names = load_data()
    
    print("\n层次聚类通过合并或分裂聚类来构建聚类的层次结构。")
    print("有两种主要方法：凝聚式（自下而上）和分裂式（自上而下）。")
    
    # 使用不同的链接方法
    linkages = ['ward', 'complete', 'average', 'single']
    n_clusters = 3
    
    plt.figure(figsize=(20, 15))
    
    for i, linkage in enumerate(linkages):
        # 创建层次聚类模型
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            compute_distances=True  # 计算距离以便绘制树状图
        )
        
        # 训练模型
        labels = model.fit_predict(X)
        
        print(f"\n使用{linkage}链接方法:")
        evaluate_clustering(X, labels, y)
        
        # 可视化聚类结果
        plt.subplot(2, 2, i+1)
        plot_dendrogram(model, truncate_mode='level', p=3)
        plt.title(f'层次聚类树状图 (链接方法: {linkage})')
        plt.xlabel('样本索引或聚类')
        plt.ylabel('距离')
    
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/hierarchical_dendrograms.png")
    plt.close()
    
    print("\n层次聚类树状图已保存为 'hierarchical_dendrograms.png'")
    
    # 使用ward链接方法（通常效果最好）
    best_linkage = 'ward'
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=best_linkage
    )
    labels = model.fit_predict(X)
    
    # 可视化最佳聚类结果
    plt_hc = visualize_clusters(X, labels, title=f"层次聚类 (链接方法: {best_linkage})")
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt_hc.savefig("images/hierarchical_clusters.png")
    plt_hc.close()
    
    print("\n层次聚类结果已保存为 'hierarchical_clusters.png'")
    
    # 层次聚类的优缺点
    print("\n层次聚类的优点:")
    print("- 不需要预先指定聚类数（可以从树状图中选择）")
    print("- 可以发现任意形状的聚类")
    print("- 提供聚类的层次结构")
    print("- 结果稳定（不依赖于初始化）")
    
    print("\n层次聚类的缺点:")
    print("- 计算复杂度高，不适用于大型数据集")
    print("- 一旦合并或分裂完成，不能撤销")
    print("- 对异常值敏感")
    print("- 不同的链接方法可能产生不同的结果")


def dbscan_example():
    """
    DBSCAN聚类示例
    """
    print("\n" + "=" * 50)
    print("DBSCAN聚类".center(50))
    print("=" * 50)
    
    # 加载数据
    X, y, feature_names, target_names = load_data()
    
    print("\nDBSCAN (基于密度的带噪声的空间聚类) 是一种基于密度的聚类算法。")
    print("它可以发现任意形状的聚类，并自动识别噪声点。")
    
    # 尝试不同的参数
    eps_values = [0.3, 0.5, 0.7, 1.0]
    min_samples_values = [3, 5, 10]
    
    best_silhouette = -1
    best_params = None
    best_labels = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # 如果所有点都被标记为噪声(-1)或只有一个聚类，跳过评估
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                print(f"eps={eps}, min_samples={min_samples}: 聚类数={n_clusters}，跳过评估")
                continue
            
            # 计算轮廓系数（排除噪声点）
            if -1 in labels:
                # 创建掩码，排除噪声点
                mask = labels != -1
                if sum(mask) < 2:  # 需要至少两个非噪声点来计算轮廓系数
                    print(f"eps={eps}, min_samples={min_samples}: 非噪声点数量不足，跳过评估")
                    continue
                silhouette = silhouette_score(X[mask], labels[mask])
            else:
                silhouette = silhouette_score(X, labels)
            
            n_noise = list(labels).count(-1)
            print(f"eps={eps}, min_samples={min_samples}: 聚类数={n_clusters}, 噪声点数={n_noise}, 轮廓系数={silhouette:.4f}")
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_params = (eps, min_samples)
                best_labels = labels
    
    if best_params is None:
        print("\n未找到合适的DBSCAN参数，使用默认参数...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        best_labels = dbscan.fit_predict(X)
        best_params = (0.5, 5)
    else:
        print(f"\n最佳参数: eps={best_params[0]}, min_samples={best_params[1]}, 轮廓系数={best_silhouette:.4f}")
    
    # 评估最佳模型
    n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise = list(best_labels).count(-1)
    
    print(f"\nDBSCAN聚类结果:")
    print(f"聚类数: {n_clusters}")
    print(f"噪声点数: {n_noise} ({n_noise/len(best_labels)*100:.2f}%)")
    
    # 如果有非噪声点，评估聚类结果
    if -1 in best_labels:
        mask = best_labels != -1
        if sum(mask) >= 2:  # 需要至少两个非噪声点来计算评估指标
            evaluate_clustering(X[mask], best_labels[mask], y[mask])
    else:
        evaluate_clustering(X, best_labels, y)
    
    # 可视化聚类结果
    plt_dbscan = visualize_clusters(
        X, best_labels, 
        title=f"DBSCAN聚类 (eps={best_params[0]}, min_samples={best_params[1]})"
    )
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt_dbscan.savefig("images/dbscan_clusters.png")
    plt_dbscan.close()
    
    print("\nDBSCAN聚类结果已保存为 'dbscan_clusters.png'")
    
    # DBSCAN的优缺点
    print("\nDBSCAN的优点:")
    print("- 不需要预先指定聚类数")
    print("- 可以发现任意形状的聚类")
    print("- 能够识别噪声点")
    print("- 对异常值不敏感")
    print("- 只需要两个参数")
    
    print("\nDBSCAN的缺点:")
    print("- 对参数敏感（eps和min_samples）")
    print("- 在密度变化很大的数据集上表现不佳")
    print("- 在高维空间中可能效果不好")
    print("- 当聚类的密度差异很大时可能无法正确分组")


def main():
    """主函数，运行所有示例"""
    # 设置matplotlib显示中文
    set_matplotlib_chinese()
    kmeans_example()
    hierarchical_clustering_example()
    dbscan_example()


if __name__ == "__main__":
    main()