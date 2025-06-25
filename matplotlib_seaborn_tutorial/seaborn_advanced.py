"""
Seaborn高级教程

本模块涵盖Seaborn的高级特性，包括：
1. 复杂统计可视化
2. 多变量分析
3. 回归分析图
4. 高级分类数据可视化
5. 自定义调色板和样式
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib_seaborn_tutorial import config

# 设置中文显示
config.setup_chinese_display()

# 设置随机种子，确保结果可重现
np.random.seed(42)

def create_advanced_sample_data():
    """
    创建用于高级示例的数据集
    """
    # 创建一个包含多个变量的DataFrame
    n = 500

    # 创建一些相关的连续变量
    x = np.random.normal(0, 1, n)
    y = x * 0.8 + np.random.normal(0, 0.5, n)
    z = x * 0.3 + y * 0.4 + np.random.normal(0, 0.3, n)

    # 创建一些分类变量
    categories = ['A', 'B', 'C', 'D']
    cat1 = np.random.choice(categories, n)
    cat2 = np.random.choice(['Group 1', 'Group 2', 'Group 3'], n)

    # 创建一些时间序列数据
    dates = pd.date_range('2020-01-01', periods=n)
    time_values = np.cumsum(np.random.normal(0, 1, n))

    # 创建一些多项式关系
    poly_x = np.linspace(-3, 3, n)
    poly_y = 0.5 * poly_x**2 + poly_x + np.random.normal(0, 1, n)

    # 创建DataFrame
    data = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'category1': cat1,
        'category2': cat2,
        'date': dates,
        'time_value': time_values,
        'poly_x': poly_x,
        'poly_y': poly_y
    })

    # 添加一些缺失值
    mask = np.random.random(n) < 0.05
    data.loc[mask, 'y'] = np.nan

    # 添加一些异常值
    outlier_mask = np.random.random(n) < 0.02
    data.loc[outlier_mask, 'z'] = data.loc[outlier_mask, 'z'] * 5

    return data

def advanced_regression_plots(data):
    """
    高级回归分析图示例
    """
    print("\n" + "=" * 50)
    print("高级回归分析图")
    print("=" * 50)

    # 1. 线性回归图
    plt.figure(figsize=(10, 8))
    sns.regplot(data=data, x='x', y='y', scatter_kws={'alpha': 0.5})
    plt.title('简单线性回归')
    plt.tight_layout()
    plt.show()

    # 2. 多项式回归
    plt.figure(figsize=(10, 8))
    sns.regplot(data=data, x='poly_x', y='poly_y', scatter_kws={'alpha': 0.5},
               order=2, line_kws={'color': 'red'})
    plt.title('多项式回归 (二次)')
    plt.tight_layout()
    plt.show()

    # 3. 分组回归
    plt.figure(figsize=(12, 8))
    sns.lmplot(data=data, x='x', y='y', hue='category2',
              scatter_kws={'alpha': 0.5}, height=8, aspect=1.2)
    plt.title('分组回归分析')
    plt.tight_layout()
    plt.show()

    # 4. 回归残差图
    plt.figure(figsize=(10, 8))
    sns.residplot(data=data, x='x', y='y', scatter_kws={'alpha': 0.5})
    plt.title('回归残差图')
    plt.tight_layout()
    plt.show()

    # 5. 分面回归图
    g = sns.lmplot(data=data, x='x', y='y', col='category1', hue='category2',
                  col_wrap=2, height=4, aspect=1.2, scatter_kws={'alpha': 0.5})
    g.fig.suptitle('分面回归图', y=1.05)
    plt.tight_layout()
    plt.show()

    # 6. 回归边际分布图
    plt.figure(figsize=(10, 8))
    sns.jointplot(data=data, x='x', y='y', kind='reg', height=8,
                 joint_kws={'scatter_kws': {'alpha': 0.5}})
    plt.suptitle('回归边际分布图', y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n高级回归分析图的关键点:")
    print("1. regplot(): 基本回归图，可以拟合多项式")
    print("2. lmplot(): 分组回归图，支持分面")
    print("3. residplot(): 残差图，用于检查回归假设")
    print("4. jointplot(kind='reg'): 带边际分布的回归图")
    print("5. 可以通过order参数拟合多项式回归")
    print("6. 可以通过hue参数按类别分组")

def multivariate_analysis(data):
    """
    多变量分析示例
    """
    print("\n" + "=" * 50)
    print("多变量分析")
    print("=" * 50)

    # 1. 成对关系图
    subset_data = data[['x', 'y', 'z', 'category2']].sample(200)
    g = sns.pairplot(subset_data, hue='category2', diag_kind='kde', height=2.5,
                    plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.6})
    g.fig.suptitle('成对关系图', y=1.02)
    plt.tight_layout()
    plt.show()

    # 2. 相关矩阵热图
    corr = data[['x', 'y', 'z', 'poly_x', 'poly_y', 'time_value']].corr()
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 创建上三角掩码
    sns.heatmap(corr, annot=True, cmap='coolwarm', mask=mask,
               vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    plt.title('相关矩阵热图')
    plt.tight_layout()
    plt.show()

    # 3. 聚类热图
    plt.figure(figsize=(12, 10))
    sns.clustermap(corr, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                  linewidths=.5, figsize=(12, 10), annot=True)
    plt.title('聚类热图')
    plt.tight_layout()
    plt.show()

    # 4. 多变量KDE图
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=data, x='x', y='y', hue='category2', fill=True, alpha=0.5)
    plt.title('多变量KDE图')
    plt.tight_layout()
    plt.show()

    # 5. 三维散点图（使用Matplotlib的3D功能）
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同类别设置不同颜色
    categories = data['category2'].unique()
    colors = sns.color_palette('husl', len(categories))

    for i, category in enumerate(categories):
        subset = data[data['category2'] == category]
        ax.scatter(subset['x'], subset['y'], subset['z'],
                  c=[colors[i]], label=category, alpha=0.6)

    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.legend()
    ax.set_title('三维散点图')
    plt.tight_layout()
    plt.show()

    print("\n多变量分析的关键点:")
    print("1. pairplot(): 创建变量对之间的关系图矩阵")
    print("2. heatmap(): 创建相关矩阵热图")
    print("3. clustermap(): 创建聚类热图，自动对变量进行聚类")
    print("4. kdeplot(): 创建多变量核密度估计图")
    print("5. 3D散点图可以同时可视化三个连续变量")

def time_series_analysis(data):
    """
    时间序列分析示例
    """
    print("\n" + "=" * 50)
    print("时间序列分析")
    print("=" * 50)

    # 准备时间序列数据
    time_data = data[['date', 'time_value', 'category2']].copy()

    # 设置日期为索引
    time_data.set_index('date', inplace=True)

    # 1. 基本时间序列图
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=time_data, x=time_data.index, y='time_value')
    plt.title('基本时间序列图')
    plt.xlabel('日期')
    plt.ylabel('值')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. 分组时间序列图
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=time_data, x=time_data.index, y='time_value', hue='category2')
    plt.title('分组时间序列图')
    plt.xlabel('日期')
    plt.ylabel('值')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. 时间序列分解
    # 重采样到月度数据
    monthly_data = time_data['time_value'].resample('M').mean()

    # 创建一个简单的趋势和季节性组件
    trend = np.linspace(0, 5, len(monthly_data))
    seasonal = 2 * np.sin(np.linspace(0, 2*np.pi*4, len(monthly_data)))
    monthly_data = pd.Series(trend + seasonal + np.random.normal(0, 0.5, len(monthly_data)),
                           index=monthly_data.index)

    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.plot(monthly_data)
    plt.title('原始时间序列')

    plt.subplot(312)
    plt.plot(trend)
    plt.title('趋势组件')

    plt.subplot(313)
    plt.plot(seasonal)
    plt.title('季节性组件')

    plt.tight_layout()
    plt.show()

    # 4. 滚动统计
    window = 30  # 30天窗口
    rolling_mean = time_data['time_value'].rolling(window=window).mean()
    rolling_std = time_data['time_value'].rolling(window=window).std()

    plt.figure(figsize=(12, 6))
    plt.plot(time_data.index, time_data['time_value'], label='原始数据')
    plt.plot(rolling_mean.index, rolling_mean, label=f'{window}天滚动平均')
    plt.plot(rolling_std.index, rolling_std, label=f'{window}天滚动标准差')
    plt.title('滚动统计')
    plt.xlabel('日期')
    plt.ylabel('值')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("\n时间序列分析的关键点:")
    print("1. lineplot(): 绘制时间序列图")
    print("2. 可以使用hue参数按类别分组")
    print("3. 可以进行时间序列分解，分离趋势和季节性")
    print("4. rolling()方法计算滚动统计量")
    print("5. resample()方法进行时间序列重采样")

def advanced_categorical_plots(data):
    """
    高级分类数据可视化示例
    """
    print("\n" + "=" * 50)
    print("高级分类数据可视化")
    print("=" * 50)

    # 准备分类数据
    cat_data = data.copy()

    # 为每个类别计算统计量
    cat_stats = cat_data.groupby(['category1', 'category2']).agg({
        'x': ['mean', 'std'],
        'y': ['mean', 'std'],
        'z': ['mean', 'std', 'count']
    }).reset_index()

    # 展平多级列索引
    cat_stats.columns = ['_'.join(col).strip('_') for col in cat_stats.columns.values]

    # 1. 复杂热图
    # 创建交叉表
    cross_tab = pd.crosstab(cat_data['category1'], cat_data['category2'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='d', cbar_kws={'label': '频率'})
    plt.title('类别交叉表热图')
    plt.tight_layout()
    plt.show()
    
    # 2. 分面分类图
    g = sns.catplot(data=cat_data, kind='box', x='category1', y='z', 
                   col='category2', height=4, aspect=1.2)
    g.fig.suptitle('分面箱线图', y=1.05)
    plt.tight_layout()
    plt.show()
    
    # 3. 复杂小提琴图
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=cat_data, x='category1', y='z', hue='category2',
                  split=True, inner='quartile')
    plt.title('分组小提琴图')
    plt.tight_layout()
    plt.show()
    
    # 4. 点图和误差线
    plt.figure(figsize=(12, 6))
    sns.pointplot(data=cat_stats, x='category1', y='x_mean', hue='category2',
                 capsize=0.1, errwidth=1.5, errorbar='sd')
    plt.title('均值和标准差点图')
    plt.tight_layout()
    plt.show()
    
    # 5. 复杂条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(data=cat_stats, x='category1', y='z_count', hue='category2',
               alpha=0.8)
    plt.title('分组计数条形图')
    plt.tight_layout()
    plt.show()
    
    print("\n高级分类数据可视化的关键点:")
    print("1. 可以使用交叉表和热图展示类别关系")
    print("2. catplot支持多种分面可视化")
    print("3. violinplot可以展示分布的形状")
    print("4. pointplot可以展示均值和误差")
    print("5. 可以组合多种图表类型来展示不同的统计特征")

def custom_palettes_and_styles():
    """
    自定义调色板和样式示例
    """
    print("\n" + "=" * 50)
    print("自定义调色板和样式")
    print("=" * 50)
    
    # 创建示例数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    data = pd.DataFrame({'x': x, 'y': y})
    
    # 1. 自定义颜色调色板
    custom_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    with sns.color_palette(custom_colors):
        sns.lineplot(data=data, x='x', y='y')
    plt.title('自定义颜色')
    
    plt.subplot(132)
    with sns.color_palette("husl", 8):
        sns.lineplot(data=data, x='x', y='y')
    plt.title('HUSL调色板')
    
    plt.subplot(133)
    with sns.color_palette("cubehelix", 8):
        sns.lineplot(data=data, x='x', y='y')
    plt.title('Cubehelix调色板')
    
    plt.tight_layout()
    plt.show()
    
    # 2. 自定义样式
    styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
    fig = plt.figure(figsize=(15, 10))
    
    for i, style in enumerate(styles):
        with sns.axes_style(style):
            plt.subplot(2, 3, i+1)
            sns.lineplot(data=data, x='x', y='y')
            plt.title(f'样式: {style}')
    
    plt.tight_layout()
    plt.show()
    
    # 3. 自定义上下文
    contexts = ['paper', 'notebook', 'talk', 'poster']
    fig = plt.figure(figsize=(15, 10))
    
    for i, context in enumerate(contexts):
        with sns.plotting_context(context):
            plt.subplot(2, 2, i+1)
            sns.lineplot(data=data, x='x', y='y')
            plt.title(f'上下文: {context}')
    
    plt.tight_layout()
    plt.show()
    
    print("\n自定义调色板和样式的关键点:")
    print("1. color_palette(): 创建自定义颜色调色板")
    print("2. axes_style(): 设置图表样式")
    print("3. plotting_context(): 设置图表上下文")
    print("4. 可以使用with语句临时设置样式")
    print("5. 支持多种预定义的调色板和样式")

def run_example():
    """
    运行所有示例
    """
    # 创建示例数据
    data = create_advanced_sample_data()
    
    # 高级回归分析图
    advanced_regression_plots(data)
    
    # 多变量分析
    multivariate_analysis(data)
    
    # 时间序列分析
    time_series_analysis(data)
    
    # 高级分类数据可视化
    advanced_categorical_plots(data)
    
    # 自定义调色板和样式
    custom_palettes_and_styles()
    
    print("\n" + "=" * 50)
    print("Seaborn高级教程完成！")
    print("=" * 50)

if __name__ == "__main__":
    run_example()