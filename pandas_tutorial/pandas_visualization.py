"""
Pandas数据可视化教程
本模块介绍Pandas库的数据可视化功能，包括基本绘图、统计图表、多子图等内容。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def basic_plotting():
    """
    演示Pandas的基本绘图功能
    包括线图、柱状图、散点图等
    """
    print("="*50)
    print("Pandas基本绘图")
    print("="*50)

    # 创建示例数据
    print("\n创建示例数据:")
    dates = pd.date_range('2023-01-01', periods=12, freq='M')
    df = pd.DataFrame({
        'A': np.random.randn(12).cumsum(),
        'B': np.random.randn(12).cumsum(),
        'C': np.random.randn(12).cumsum()
    }, index=dates)
    print(df)

    # 1. 线图
    print("\n1. 线图:")
    print("执行以下代码查看线图:")
    print("plt.figure(figsize=(10, 6))")
    # 创建一个新的图形窗口，设置其大小为宽10英寸、高6英寸
    plt.figure(figsize=(10, 6))
    # 使用 pandas 的 plot 方法对 DataFrame（df）绘制线图，设置图表标题为“线图示例”
    df.plot(title='线图示例')
    # 添加图例，并自动选择最佳位置显示
    plt.legend(loc='best')
    # 显示网格线，方便查看数据点的位置
    plt.grid(True)
    # 显示绘制好的图形窗口
    plt.show()

    # 2. 柱状图
    print("\n2. 柱状图:")
    print("执行以下代码查看柱状图:")
    print("plt.figure(figsize=(10, 6))")
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', title='柱状图示例')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # 3. 散点图
    print("\n3. 散点图:")
    print("执行以下代码查看散点图:")
    print("plt.figure(figsize=(10, 6))")
    plt.figure(figsize=(10, 6))
    df.plot(kind='scatter', x='A', y='B', title='散点图示例')
    plt.grid(True)
    plt.show()

    # 4. 直方图
    print("\n4. 直方图:")
    print("执行以下代码查看直方图:")
    print("plt.figure(figsize=(10, 6))")
    plt.figure(figsize=(10, 6))
    df['A'].plot(kind='hist', bins=20, title='直方图示例')
    plt.grid(True)
    plt.show()

    # 5. 箱线图
    print("\n5. 箱线图:")
    print("执行以下代码查看箱线图:")
    print("plt.figure(figsize=(10, 6))")
    plt.figure(figsize=(10, 6))
    df.boxplot()
    plt.title('箱线图示例')
    plt.grid(True)
    plt.show()

def statistical_plotting():
    """
    演示Pandas的统计图表功能
    包括密度图、相关性热图、成对关系图等
    """
    print("\n"+"="*50)
    print("Pandas统计图表")
    print("="*50)

    # 创建示例数据
    print("\n创建示例数据:")
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 1000),
        'B': np.random.normal(2, 1.5, 1000),
        'C': np.random.normal(-1, 2, 1000),
        'D': np.random.normal(3, 0.5, 1000)
    })
    print(df.describe())

    # 1. 密度图
    print("\n1. 密度图:")
    print("执行以下代码查看密度图:")
    print("plt.figure(figsize=(10, 6))")
    plt.figure(figsize=(10, 6))
    df.plot(kind='density', title='密度图示例')
    plt.grid(True)
    plt.show()

    # 2. 相关性热图
    print("\n2. 相关性热图:")
    print("执行以下代码查看相关性热图:")
    print("plt.figure(figsize=(8, 6))")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('相关性热图示例')
    plt.show()

    # 3. 成对关系图
    print("\n3. 成对关系图:")
    print("执行以下代码查看成对关系图:")
    print("sns.pairplot(df)")
    sns.pairplot(df)
    plt.show()

    # 4. 小提琴图
    print("\n4. 小提琴图:")
    print("执行以下代码查看小提琴图:")
    print("plt.figure(figsize=(10, 6))")
    plt.figure(figsize=(10, 6))
    df.plot(kind='violin', title='小提琴图示例')
    plt.grid(True)
    plt.show()

    # 5. KDE图
    print("\n5. KDE图:")
    print("执行以下代码查看KDE图:")
    print("plt.figure(figsize=(10, 6))")
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        sns.kdeplot(data=df[column], label=column)
    plt.title('KDE图示例')
    plt.legend()
    plt.grid(True)
    plt.show()

def subplots_demo():
    """
    演示Pandas的多子图功能
    包括不同类型的子图布局和组合
    """
    print("\n"+"="*50)
    print("Pandas多子图示例")
    print("="*50)

    # 创建示例数据
    print("\n创建示例数据:")
    dates = pd.date_range('2023-01-01', periods=100)
    df = pd.DataFrame({
        'A': np.random.randn(100).cumsum(),
        'B': np.random.randn(100).cumsum(),
        'C': np.random.randn(100).cumsum(),
        'D': np.random.randn(100).cumsum()
    }, index=dates)
    print(df.head())

    # 1. 基本子图布局
    print("\n1. 基本子图布局:")
    print("执行以下代码查看基本子图布局:")
    print("fig, axes = plt.subplots(2, 2, figsize=(12, 8))")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 第一个子图：线图
    df.plot(ax=axes[0, 0], title='线图')
    axes[0, 0].grid(True)

    # 第二个子图：柱状图
    df.iloc[-10:].plot(kind='bar', ax=axes[0, 1], title='柱状图')
    axes[0, 1].grid(True)

    # 第三个子图：散点图
    df.plot(kind='scatter', x='A', y='B', ax=axes[1, 0], title='散点图')
    axes[1, 0].grid(True)

    # 第四个子图：箱线图
    df.boxplot(ax=axes[1, 1])
    axes[1, 1].set_title('箱线图')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # 2. 不同大小的子图
    print("\n2. 不同大小的子图:")
    print("执行以下代码查看不同大小的子图:")
    print("fig = plt.figure(figsize=(12, 8))")
    fig = plt.figure(figsize=(12, 8))

    # 创建网格
    gs = fig.add_gridspec(2, 2)

    # 第一个子图占据左半部分
    ax1 = fig.add_subplot(gs[:, 0])
    df.plot(ax=ax1, title='左半部分：线图')
    ax1.grid(True)

    # 右上角子图
    ax2 = fig.add_subplot(gs[0, 1])
    df.iloc[-10:].plot(kind='bar', ax=ax2, title='右上角：柱状图')
    ax2.grid(True)

    # 右下角子图
    ax3 = fig.add_subplot(gs[1, 1])
    df.plot(kind='scatter', x='A', y='B', ax=ax3, title='右下角：散点图')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # 3. 组合图表
    print("\n3. 组合图表:")
    print("执行以下代码查看组合图表:")
    print("fig, ax1 = plt.subplots(figsize=(10, 6))")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制线图
    color = 'tab:blue'
    ax1.set_xlabel('日期')
    ax1.set_ylabel('A值', color=color)
    ax1.plot(df.index, df['A'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    # 创建双轴
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('B值', color=color)
    ax2.plot(df.index, df['B'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('双轴组合图表示例')
    plt.show()

def main():
    """
    主函数，按顺序运行所有示例
    """
    print("="*50)
    print("Pandas数据可视化教程")
    print("="*50)
    print("\n本教程将介绍以下内容：")
    print("1. 基本绘图")
    print("2. 统计图表")
    print("3. 多子图示例")

    input("\n按Enter键开始演示...")

    # 设置matplotlib样式
    # 使用有效的matplotlib样式名称
    plt.style.use('seaborn-v0_8')  # 新版本中seaborn样式的正确引用方式

    # 运行所有示例
    basic_plotting()
    statistical_plotting()
    subplots_demo()

    print("\n"+"="*50)
    print("Pandas数据可视化教程完成！")
    print("="*50)

if __name__ == "__main__":
    main()
