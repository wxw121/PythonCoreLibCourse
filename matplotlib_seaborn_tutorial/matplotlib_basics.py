"""
Matplotlib基础教程

本模块涵盖Matplotlib的基础知识，包括：
1. Matplotlib架构和基本概念
2. 基本图表类型（线图、散点图、柱状图等）
3. 图表组件（标题、标签、图例等）
4. 子图和布局
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import matplotlib as mpl
from matplotlib_seaborn_tutorial import config

# 设置中文显示
config.setup_chinese_display()

def intro_to_matplotlib():
    """
    Matplotlib简介和架构
    """
    print("=" * 50)
    print("Matplotlib简介")
    print("=" * 50)
    print("Matplotlib是Python最流行的数据可视化库之一，提供了类似MATLAB的绘图API。")
    print("它由以下几个关键组件组成：")
    print("1. Figure: 整个图形窗口")
    print("2. Axes: 图形中的单个绘图区域")
    print("3. Axis: 坐标轴")
    print("4. Artist: 图形中的所有元素（线、文本、填充区域等）")
    print("\n我们将通过实例来学习这些概念。")
    
    # 显示Matplotlib的版本
    print(f"\nMatplotlib版本: {mpl.__version__}")
    
    input("\n按回车键继续...")

def basic_line_plot():
    """
    基本线图示例
    """
    print("\n" + "=" * 50)
    print("基本线图")
    print("=" * 50)
    
    # 创建数据
    x = np.linspace(0, 10, 100)  # 0到10之间的100个点
    print(f"x: {x}")
    y1 = np.sin(x)               # 正弦函数
    y2 = np.cos(x)               # 余弦函数
    
    # 创建图形和子图
    # plt.figure() 创建一个新的图形对象
    # figsize参数设置图形的宽度和高度（单位为英寸）
    plt.figure(figsize=(10, 6))
    
    # 绘制线图
    # 'b-'表示蓝色实线，'r--'表示红色虚线
    # linewidth设置线宽，label设置图例标签
    plt.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
    plt.plot(x, y2, 'r--', linewidth=2, label='cos(x)')
    
    # 添加标题和标签
    plt.title('正弦和余弦函数', fontsize=16, fontweight='bold')
    plt.xlabel('x值', fontsize=12)
    plt.ylabel('y值', fontsize=12)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(loc='best', fontsize=12)
    
    # 设置x轴和y轴的范围
    plt.xlim(0, 10)
    plt.ylim(-1.5, 1.5)

    # 显示图形
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()

    print("\n基本线图的关键点:")
    print("1. plt.figure(): 创建新的图形")
    print("2. plt.plot(): 绘制线图")
    print("3. 使用标题、标签和图例增强可读性")
    print("4. plt.grid(): 添加网格线")
    print("5. plt.xlim()/plt.ylim(): 设置坐标轴范围")
    print("6. plt.tight_layout(): 优化布局")
    print("7. plt.show(): 显示图形")

def scatter_plot():
    """
    散点图示例
    """
    print("\n" + "=" * 50)
    print("散点图")
    print("=" * 50)

    # 创建随机数据
    np.random.seed(42)  # 设置随机种子，确保结果可重现
    x = np.random.rand(50) * 10  # 50个0到10之间的随机数
    y = 2 * x + 1 + np.random.randn(50) * 2  # 线性关系加噪声

    # 创建第三个变量，用于颜色映射
    colors = np.random.rand(50)  # 50个0到1之间的随机数
    sizes = np.random.rand(50) * 200 + 50  # 点的大小变化

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制散点图
    # c参数设置颜色映射，s参数设置点的大小，alpha设置透明度
    scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.7,
                         cmap='viridis', edgecolors='k', linewidths=1)

    # 添加标题和标签
    plt.title('散点图示例', fontsize=16)
    plt.xlabel('X轴', fontsize=12)
    plt.ylabel('Y轴', fontsize=12)

    # 添加颜色条
    plt.colorbar(scatter, label='颜色值')

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)

    # 显示图形
    plt.tight_layout()
    plt.show()

    print("\n散点图的关键点:")
    print("1. plt.scatter(): 创建散点图")
    print("2. 可以通过c参数映射颜色")
    print("3. 可以通过s参数控制点的大小")
    print("4. cmap参数设置颜色映射方案")
    print("5. plt.colorbar(): 添加颜色条")

def bar_chart():
    """
    柱状图示例
    """
    print("\n" + "=" * 50)
    print("柱状图")
    print("=" * 50)

    # 创建数据
    categories = ['A', 'B', 'C', 'D', 'E']
    values1 = [5, 7, 3, 8, 6]
    values2 = [3, 5, 2, 6, 4]

    # 计算柱状图的位置
    x = np.arange(len(categories))
    width = 0.35  # 柱子的宽度

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制柱状图
    bars1 = plt.bar(x - width/2, values1, width, label='组1', color='skyblue', edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + width/2, values2, width, label='组2', color='lightcoral', edgecolor='black', linewidth=1)

    # 添加数据标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{height}', ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)

    # 添加标题和标签
    plt.title('分组柱状图示例', fontsize=16)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('值', fontsize=12)

    # 设置x轴刻度和标签
    plt.xticks(x, categories)

    # 添加图例
    plt.legend()

    # 添加网格线（仅y轴）
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 显示图形
    plt.tight_layout()
    plt.show()

    print("\n柱状图的关键点:")
    print("1. plt.bar(): 创建柱状图")
    print("2. 通过调整位置可以创建分组柱状图")
    print("3. 可以使用plt.text()添加数据标签")
    print("4. plt.xticks(): 设置x轴刻度和标签")

def histogram():
    """
    直方图示例
    """
    print("\n" + "=" * 50)
    print("直方图")
    print("=" * 50)

    # 创建随机数据
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)  # 均值为0，标准差为1的正态分布
    data2 = np.random.normal(3, 1.5, 1000)  # 均值为3，标准差为1.5的正态分布

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制直方图
    # bins设置箱子数量，alpha设置透明度，density=True表示归一化
    plt.hist(data1, bins=30, alpha=0.7, color='skyblue', edgecolor='black',
             linewidth=1, label='分布1', density=True)
    plt.hist(data2, bins=30, alpha=0.7, color='lightcoral', edgecolor='black',
             linewidth=1, label='分布2', density=True)

    # 添加标题和标签
    plt.title('直方图示例', fontsize=16)
    plt.xlabel('值', fontsize=12)
    plt.ylabel('密度', fontsize=12)

    # 添加图例
    plt.legend()

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.3)

    # 显示图形
    plt.tight_layout()
    plt.show()

    print("\n直方图的关键点:")
    print("1. plt.hist(): 创建直方图")
    print("2. bins参数控制箱子数量")
    print("3. density=True将频率归一化为密度")
    print("4. 可以在同一图上绘制多个直方图进行比较")

def pie_chart():
    """
    饼图示例
    """
    print("\n" + "=" * 50)
    print("饼图")
    print("=" * 50)

    # 创建数据
    labels = ['A', 'B', 'C', 'D', 'E']
    sizes = [15, 30, 25, 10, 20]
    colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'lightpink']
    explode = (0.1, 0, 0, 0, 0)  # 突出第一个扇形

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 绘制饼图
    # autopct参数设置显示百分比的格式
    # startangle参数设置起始角度
    # shadow参数设置是否有阴影
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)

    # 添加标题
    plt.title('饼图示例', fontsize=16)

    # 确保饼图是圆形的
    plt.axis('equal')

    # 显示图形
    plt.tight_layout()
    plt.show()

    print("\n饼图的关键点:")
    print("1. plt.pie(): 创建饼图")
    print("2. explode参数可以突出显示某些扇形")
    print("3. autopct参数控制百分比的显示格式")
    print("4. plt.axis('equal')确保饼图是圆形的")

def subplots_demo():
    """
    子图示例
    """
    print("\n" + "=" * 50)
    print("子图布局")
    print("=" * 50)

    # 创建数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = x**2

    # 创建2x2的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 绘制第一个子图 (0,0) - 线图
    axes[0, 0].plot(x, y1, 'b-', linewidth=2)
    axes[0, 0].set_title('正弦函数')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('sin(x)')
    axes[0, 0].grid(True)

    # 绘制第二个子图 (0,1) - 余弦图
    axes[0, 1].plot(x, y2, 'r-', linewidth=2)
    axes[0, 1].set_title('余弦函数')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('cos(x)')
    axes[0, 1].grid(True)

    # 绘制第三个子图 (1,0) - 正切图
    # 限制y轴范围，因为正切函数有无穷大值
    axes[1, 0].plot(x, y3, 'g-', linewidth=2)
    axes[1, 0].set_title('正切函数')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('tan(x)')
    axes[1, 0].set_ylim(-5, 5)  # 限制y轴范围
    axes[1, 0].grid(True)
    
    # 绘制第四个子图 (1,1) - 二次函数
    axes[1, 1].plot(x, y4, 'm-', linewidth=2)
    axes[1, 1].set_title('二次函数')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('x^2')
    axes[1, 1].grid(True)
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    print("\n子图的关键点:")
    print("1. plt.subplots(): 创建子图网格")
    print("2. 使用axes[i, j]访问特定的子图")
    print("3. 每个子图都是一个独立的Axes对象，有自己的方法")
    print("4. tight_layout()自动调整子图布局")

def custom_layout():
    """
    自定义布局示例
    """
    print("\n" + "=" * 50)
    print("自定义布局")
    print("=" * 50)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建自定义布局的子图
    # 参数表示[左, 底, 宽, 高]，范围是0到1
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # 第一行，跨越两列
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)  # 右上角，跨越两行
    ax3 = plt.subplot2grid((3, 3), (1, 0))             # 中间行，第一列
    ax4 = plt.subplot2grid((3, 3), (1, 1))             # 中间行，第二列
    ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)  # 底行，跨越三列
    
    # 在每个子图中添加一些内容
    ax1.text(0.5, 0.5, 'Subplot 1', ha='center', va='center', fontsize=12)
    ax2.text(0.5, 0.5, 'Subplot 2', ha='center', va='center', fontsize=12)
    ax3.text(0.5, 0.5, 'Subplot 3', ha='center', va='center', fontsize=12)
    ax4.text(0.5, 0.5, 'Subplot 4', ha='center', va='center', fontsize=12)
    ax5.text(0.5, 0.5, 'Subplot 5', ha='center', va='center', fontsize=12)
    
    # 设置每个子图的标题
    ax1.set_title('跨越两列')
    ax2.set_title('跨越两行')
    ax3.set_title('单个单元格')
    ax4.set_title('单个单元格')
    ax5.set_title('跨越三列')
    
    # 为每个子图添加网格
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.grid(True, linestyle='--', alpha=0.7)
        # 移除刻度标签，使显示更清晰
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    print("\n自定义布局的关键点:")
    print("1. plt.subplot2grid(): 创建自定义网格布局")
    print("2. 可以指定子图的位置和跨度")
    print("3. 灵活性高，可以创建复杂的布局")

def figure_and_axes():
    """
    Figure和Axes对象的详细说明
    """
    print("\n" + "=" * 50)
    print("Figure和Axes对象")
    print("=" * 50)
    
    # 创建Figure对象和Axes对象
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 创建数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # 使用Axes对象的方法绘图
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    
    # 设置标题和标签
    ax.set_title('使用Axes对象的方法', fontsize=16)
    ax.set_xlabel('x值', fontsize=12)
    ax.set_ylabel('y值', fontsize=12)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    ax.legend(loc='best')
    
    # 设置x轴和y轴的范围
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.5)
    
    # 添加文本注释
    ax.text(5, 0.5, 'Axes对象提供了\n大多数绘图功能', 
            ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # 显示图形
    plt.tight_layout()
    plt.show()
    
    print("\nFigure和Axes对象的关键点:")
    print("1. Figure是整个图形窗口")
    print("2. Axes是图形中的单个绘图区域")
    print("3. 大多数绘图命令都可以通过Axes对象调用")
    print("4. 面向对象的接口(ax.plot())比pyplot接口(plt.plot())更灵活")
    print("5. 对于复杂的图形，推荐使用面向对象的接口")

def annotations_and_text():
    """
    注释和文本示例
    """
    print("\n" + "=" * 50)
    print("注释和文本")
    print("=" * 50)
    
    # 创建数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制线图
    ax.plot(x, y, 'b-', linewidth=2)
    
    # 添加标题和标签
    ax.set_title('文本和注释示例', fontsize=16)
    ax.set_xlabel('x值', fontsize=12)
    ax.set_ylabel('y值', fontsize=12)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加简单文本
    ax.text(1, 0.8, '简单文本', fontsize=12)
    
    # 添加带背景的文本
    ax.text(3, 0.8, '带背景的文本', fontsize=12,
           bbox=dict(facecolor='yellow', alpha=0.5, boxstyle='round'))
    
    # 添加带箭头的注释
    ax.annotate('局部最大值', xy=(4.7, 0.99), xytext=(5.5, 0.7),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
               fontsize=12)
    
    # 添加数学公式
    ax.text(7, 0.8, r'$\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} x^{2n+1}$',
           fontsize=14)
    
    # 显示图形
    plt.tight_layout()
    plt.show()
    
    print("\n注释和文本的关键点:")
    print("1. ax.text(): 在图形上添加文本")
    print("2. ax.annotate(): 添加带箭头的注释")
    print("3. 可以使用LaTeX语法添加数学公式")
    print("4. 可以自定义文本的外观（颜色、大小、背景等）")

def run_example():
    """
    运行所有示例
    """
    # 介绍Matplotlib
    intro_to_matplotlib()
    
    # 基本图表类型
    basic_line_plot()
    scatter_plot()
    bar_chart()
    histogram()
    pie_chart()
    
    # 子图和布局
    subplots_demo()
    custom_layout()
    
    # 高级概念
    figure_and_axes()
    annotations_and_text()
    
    print("\n" + "=" * 50)
    print("Matplotlib基础教程完成！")
    print("=" * 50)

if __name__ == "__main__":
    run_example()