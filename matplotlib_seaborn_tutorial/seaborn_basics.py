"""
Seaborn基础教程

本模块涵盖Seaborn的基础知识，包括：
1. Seaborn简介和与Matplotlib的关系
2. 基本统计图表
3. 分布图
4. 关系图
5. 分类图
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib_seaborn_tutorial import config

# 设置中文显示
config.setup_chinese_display()

# 设置随机种子，确保结果可重现
np.random.seed(42)

def intro_to_seaborn():
    """
    Seaborn简介
    """
    print("=" * 50)
    print("Seaborn简介")
    print("=" * 50)
    print("Seaborn是基于Matplotlib的高级统计可视化库，提供了更美观的默认样式和更高级的统计图表。")
    print("主要优势包括：")
    print("1. 提供了更美观的默认样式")
    print("2. 内置了多种统计图表")
    print("3. 与Pandas数据结构紧密集成")
    print("4. 自动处理分类变量")
    print("5. 内置多种颜色主题")

    # 显示Seaborn的版本
    print(f"\nSeaborn版本: {sns.__version__}")

    # 显示可用的样式和调色板
    print("\n可用的样式:")
    print(sns.axes_style().keys())

    print("\n可用的调色板:")
    print(", ".join(sns.color_palette().as_hex()))

    input("\n按回车键继续...")

def create_sample_data():
    """
    创建示例数据集
    """
    # 创建一个包含多个变量的DataFrame
    n = 200

    # 创建一些相关的连续变量
    data = pd.DataFrame({
        'x': np.random.normal(0, 1, n),
        'y': np.random.normal(0, 1, n),
        'z': np.random.normal(0, 1, n)
    })

    # 添加一些分类变量
    data['category1'] = np.random.choice(['A', 'B', 'C'], n)
    data['category2'] = np.random.choice(['Group 1', 'Group 2'], n)

    # 添加一些有关系的变量
    data['correlated'] = data['x'] * 0.8 + np.random.normal(0, 0.5, n)
    data['size'] = np.abs(data['z']) * 100

    return data

def seaborn_themes():
    """
    Seaborn主题和样式
    """
    print("\n" + "=" * 50)
    print("Seaborn主题和样式")
    print("=" * 50)

    # 创建示例数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # 创建一个2x3的子图网格来展示不同的样式
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 可用的样式
    styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

    # 在每个子图中使用不同的样式
    for i, style in enumerate(styles):
        row, col = i // 3, i % 3
        with sns.axes_style(style):
            ax = axes[row, col]
            ax.plot(x, y)
            ax.set_title(f"样式: {style}")

    # 最后一个子图展示调色板
    with sns.color_palette("viridis"):
        ax = axes[1, 2]
        for i in range(6):
            ax.plot(x, np.sin(x + i * 0.5))
        ax.set_title("调色板: viridis")

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    print("\nSeaborn主题和样式的关键点:")
    print("1. sns.set_style(): 设置图表样式")
    print("2. sns.set_palette(): 设置调色板")
    print("3. sns.axes_style(): 临时设置样式")
    print("4. sns.color_palette(): 创建颜色调色板")
    print("5. 可用样式: darkgrid, whitegrid, dark, white, ticks")

def distribution_plots(data):
    """
    分布图示例
    """
    print("\n" + "=" * 50)
    print("分布图")
    print("=" * 50)

    # 创建一个2x2的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 直方图
    sns.histplot(data=data, x='x', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('直方图 (histplot)')

    # 2. 核密度估计图
    sns.kdeplot(data=data, x='x', fill=True, ax=axes[0, 1])
    axes[0, 1].set_title('核密度估计图 (kdeplot)')

    # 3. 经验累积分布函数
    sns.ecdfplot(data=data, x='x', ax=axes[1, 0])
    axes[1, 0].set_title('经验累积分布函数 (ecdfplot)')

    # 4. 箱线图
    sns.boxplot(data=data, x='category1', y='x', ax=axes[1, 1])
    axes[1, 1].set_title('箱线图 (boxplot)')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    # 创建另一个图形展示更多分布图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 5. 小提琴图
    sns.violinplot(data=data, x='category1', y='x', ax=axes[0, 0])
    axes[0, 0].set_title('小提琴图 (violinplot)')

    # 6. 条形图
    sns.barplot(data=data, x='category1', y='x', ax=axes[0, 1])
    axes[0, 1].set_title('条形图 (barplot)')

    # 7. 计数图
    sns.countplot(data=data, x='category1', ax=axes[1, 0])
    axes[1, 0].set_title('计数图 (countplot)')

    # 8. 带分组的箱线图
    sns.boxplot(data=data, x='category1', y='x', hue='category2', ax=axes[1, 1])
    axes[1, 1].set_title('带分组的箱线图')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    print("\n分布图的关键点:")
    print("1. histplot(): 绘制直方图，可选择添加核密度估计")
    print("2. kdeplot(): 绘制核密度估计图")
    print("3. ecdfplot(): 绘制经验累积分布函数")
    print("4. boxplot(): 绘制箱线图")
    print("5. violinplot(): 绘制小提琴图")
    print("6. barplot(): 绘制条形图，显示平均值和置信区间")
    print("7. countplot(): 绘制计数图，显示分类变量的频率")
    print("8. 可以使用hue参数添加额外的分组维度")

def relationship_plots(data):
    """
    关系图示例
    """
    print("\n" + "=" * 50)
    print("关系图")
    print("=" * 50)

    # 创建一个2x2的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 散点图
    sns.scatterplot(data=data, x='x', y='correlated', ax=axes[0, 0])
    axes[0, 0].set_title('散点图 (scatterplot)')

    # 2. 带有色调和大小的散点图
    sns.scatterplot(data=data, x='x', y='correlated', hue='category1',
                   size='size', sizes=(20, 200), alpha=0.7, ax=axes[0, 1])
    axes[0, 1].set_title('带有色调和大小的散点图')

    # 3. 线图
    # 创建一些时间序列数据
    time_data = pd.DataFrame({
        'time': np.arange(100),
        'value': np.cumsum(np.random.randn(100)),
        'group': np.repeat(['A', 'B'], 50)
    })
    sns.lineplot(data=time_data, x='time', y='value', hue='group', ax=axes[1, 0])
    axes[1, 0].set_title('线图 (lineplot)')

    # 4. 回归图
    sns.regplot(data=data, x='x', y='correlated', ax=axes[1, 1])
    axes[1, 1].set_title('回归图 (regplot)')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    # 创建另一个图形展示更多关系图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 5. 成对关系图（在一个子图中）
    subset_data = data[['x', 'y', 'z', 'correlated']].sample(100)
    sns.pairplot(subset_data, height=2.5)
    plt.suptitle('成对关系图 (pairplot)', y=1.02)
    plt.show()

    # 6. 联合分布图
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.jointplot(data=data, x='x', y='correlated', kind='scatter', height=8)
    plt.suptitle('联合分布图 (jointplot)', y=0.95)
    plt.tight_layout()
    plt.show()

    # 7. 六边形箱图
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.jointplot(data=data, x='x', y='correlated', kind='hex', height=8)
    plt.suptitle('六边形箱图 (jointplot with kind="hex")', y=0.95)
    plt.tight_layout()
    plt.show()

    # 8. 热图
    # 创建相关矩阵
    corr_matrix = data[['x', 'y', 'z', 'correlated']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('热图 (heatmap)')
    plt.tight_layout()
    plt.show()

    print("\n关系图的关键点:")
    print("1. scatterplot(): 绘制散点图，可以添加色调和大小维度")
    print("2. lineplot(): 绘制线图，适合时间序列数据")
    print("3. regplot(): 绘制回归图，自动添加回归线")
    print("4. pairplot(): 绘制变量对之间的关系")
    print("5. jointplot(): 绘制两个变量的联合分布")
    print("6. heatmap(): 绘制热图，适合相关矩阵")

def categorical_plots(data):
    """
    分类图示例
    """
    print("\n" + "=" * 50)
    print("分类图")
    print("=" * 50)

    # 创建一个2x2的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 分类散点图
    sns.stripplot(data=data, x='category1', y='x', jitter=True, ax=axes[0, 0])
    axes[0, 0].set_title('分类散点图 (stripplot)')

    # 2. 蜂群图
    sns.swarmplot(data=data, x='category1', y='x', ax=axes[0, 1])
    axes[0, 1].set_title('蜂群图 (swarmplot)')

    # 3. 箱线图和分类散点图组合
    sns.boxplot(data=data, x='category1', y='x', ax=axes[1, 0])
    sns.stripplot(data=data, x='category1', y='x', color='black', size=3, jitter=True, alpha=0.3, ax=axes[1, 0])
    axes[1, 0].set_title('箱线图和分类散点图组合')

    # 4. 小提琴图和分类散点图组合
    sns.violinplot(data=data, x='category1', y='x', inner=None, ax=axes[1, 1])
    sns.stripplot(data=data, x='category1', y='x', color='black', size=3, jitter=True, alpha=0.3, ax=axes[1, 1])
    axes[1, 1].set_title('小提琴图和分类散点图组合')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    # 创建另一个图形展示更多分类图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 5. 分类箱线图
    sns.boxenplot(data=data, x='category1', y='x', ax=axes[0, 0])
    axes[0, 0].set_title('分类箱线图 (boxenplot)')
    
    # 6. 点图
    sns.pointplot(data=data, x='category1', y='x', hue='category2', ax=axes[0, 1])
    axes[0, 1].set_title('点图 (pointplot)')
    
    # 7. 分类条形图
    sns.barplot(data=data, x='category1', y='x', hue='category2', ax=axes[1, 0])
    axes[1, 0].set_title('分类条形图 (barplot)')
    
    # 8. 计数图
    sns.countplot(data=data, x='category1', hue='category2', ax=axes[1, 1])
    axes[1, 1].set_title('计数图 (countplot)')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    print("\n分类图的关键点:")
    print("1. stripplot(): 绘制分类散点图")
    print("2. swarmplot(): 绘制不重叠的分类散点图")
    print("3. boxplot(): 绘制箱线图")
    print("4. violinplot(): 绘制小提琴图")
    print("5. boxenplot(): 绘制增强型箱线图")
    print("6. pointplot(): 绘制点图，显示估计值和置信区间")
    print("7. barplot(): 绘制条形图，显示平均值和置信区间")
    print("8. countplot(): 绘制计数图，显示分类变量的频率")
    print("9. 可以组合多种图表类型来增强可视化效果")

def facet_grid_and_map(data):
    """
    分面网格和映射示例
    """
    print("\n" + "=" * 50)
    print("分面网格和映射")
    print("=" * 50)
    
    # 1. FacetGrid - 按类别分面
    g = sns.FacetGrid(data, col="category1", height=5, aspect=0.8)
    g.map(sns.histplot, "x", kde=True)
    g.fig.suptitle('按类别分面的直方图', y=1.05)
    plt.tight_layout()
    plt.show()
    
    # 2. FacetGrid - 按行和列分面
    g = sns.FacetGrid(data, col="category1", row="category2", height=4)
    g.map(sns.scatterplot, "x", "correlated")
    g.add_legend()
    g.fig.suptitle('按行和列分面的散点图', y=1.05)
    plt.tight_layout()
    plt.show()
    
    # 3. 使用catplot进行分类图的分面
    g = sns.catplot(data=data, kind="box", x="category1", y="x", col="category2", height=4, aspect=0.8)
    g.fig.suptitle('使用catplot的分面箱线图', y=1.05)
    plt.tight_layout()
    plt.show()
    
    # 4. 使用relplot进行关系图的分面
    g = sns.relplot(data=data, kind="scatter", x="x", y="correlated", col="category1", hue="category2", size="size", sizes=(10, 100), alpha=0.7, height=4, aspect=0.8)
    g.fig.suptitle('使用relplot的分面散点图', y=1.05)
    plt.tight_layout()
    plt.show()
    
    print("\n分面网格和映射的关键点:")
    print("1. FacetGrid: 创建分面网格")
    print("2. map(): 将绘图函数映射到网格")
    print("3. catplot(): 分类图的分面")
    print("4. relplot(): 关系图的分面")
    print("5. 分面可以按行、列或两者同时进行")

def run_example():
    """
    运行所有示例
    """
    # 介绍Seaborn
    intro_to_seaborn()
    
    # 创建示例数据
    data = create_sample_data()
    print(data)
    
    # Seaborn主题和样式
    seaborn_themes()
    
    # 分布图
    distribution_plots(data)
    
    # 关系图
    relationship_plots(data)
    
    # 分类图
    categorical_plots(data)
    
    # 分面网格和映射
    facet_grid_and_map(data)
    
    print("\n" + "=" * 50)
    print("Seaborn基础教程完成！")
    print("=" * 50)

if __name__ == "__main__":
    run_example()