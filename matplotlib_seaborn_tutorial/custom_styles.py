"""
自定义样式和主题教程

本模块涵盖如何自定义Matplotlib和Seaborn的样式和主题，包括：
1. 创建自定义样式表
2. 自定义颜色映射和调色板
3. 创建一致的可视化风格
4. 为出版物准备图形
5. 创建企业风格的可视化
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
from matplotlib_seaborn_tutorial import config

# 设置中文显示
config.setup_chinese_display()

# 设置随机种子，确保结果可重现
np.random.seed(42)

def create_sample_data():
    """
    创建用于样式示例的数据集
    """
    # 创建一些基本数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.exp(-0.1 * x)
    y4 = np.cos(x) * np.exp(-0.1 * x)
    
    # 创建一些散点数据
    scatter_x = np.random.normal(5, 1.5, 100)
    scatter_y = scatter_x * 0.5 + np.random.normal(0, 1, 100)
    
    # 创建一些分类数据
    categories = ['A', 'B', 'C', 'D']
    cat_values = np.random.normal(0, 1, 100)
    cat_groups = np.random.choice(categories, 100)
    
    # 创建一些多变量数据
    multi_data = pd.DataFrame({
        'x': np.random.normal(0, 1, 200),
        'y': np.random.normal(0, 1, 200),
        'group': np.repeat(['Group 1', 'Group 2'], 100)
    })
    
    return {
        'line_data': {
            'x': x,
            'y1': y1,
            'y2': y2,
            'y3': y3,
            'y4': y4
        },
        'scatter_data': {
            'x': scatter_x,
            'y': scatter_y
        },
        'cat_data': {
            'values': cat_values,
            'groups': cat_groups
        },
        'multi_data': multi_data
    }

def matplotlib_style_sheets():
    """
    Matplotlib内置样式表示例
    """
    print("\n" + "=" * 50)
    print("Matplotlib内置样式表")
    print("=" * 50)
    
    # 获取样本数据
    data = create_sample_data()
    x = data['line_data']['x']
    y1 = data['line_data']['y1']
    y2 = data['line_data']['y2']

    # 展示一些内置样式
    # 在新版本的Matplotlib中，样式名称可能已更改
    styles = ['default', 'classic', 'bmh', 'ggplot']
    
    # 尝试添加seaborn相关样式，如果可用的话
    seaborn_styles = []
    for style in plt.style.available:
        if 'seaborn' in style:
            seaborn_styles.append(style)
            if len(seaborn_styles) >= 2:  # 只添加最多两个seaborn样式
                break
    
    styles.extend(seaborn_styles)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, style in enumerate(styles):
        with plt.style.context(style):
            ax = axes[i]
            ax.plot(x, y1, label='sin(x)')
            ax.plot(x, y2, label='cos(x)')
            ax.set_title(f'样式: {style}')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.show()

    # 列出所有可用的样式
    print("\n可用的Matplotlib样式表:")
    print(", ".join(plt.style.available))

    print("\nMatplotlib样式表的关键点:")
    print("1. plt.style.use(): 设置全局样式")
    print("2. plt.style.context(): 临时设置样式")
    print("3. 可以组合多个样式表")
    print("4. 内置样式包括ggplot、seaborn、bmh等")

def create_custom_style_sheet():
    """
    创建自定义样式表示例
    """
    print("\n" + "=" * 50)
    print("创建自定义样式表")
    print("=" * 50)

    # 获取样本数据
    data = create_sample_data()
    x = data['line_data']['x']
    y1 = data['line_data']['y1']
    y2 = data['line_data']['y2']

    # 定义自定义样式
    custom_style = {
        # 图形和轴的大小
        'figure.figsize': (10, 6),
        'figure.dpi': 100,

        # 背景颜色和网格
        'axes.facecolor': '#f0f0f0',
        'axes.grid': True,
        'grid.color': 'white',
        'grid.linestyle': '-',
        'grid.linewidth': 1.5,

        # 线条和标记
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.prop_cycle': cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),

        # 字体和文本
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,

        # 刻度
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,

        # 图例
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'gray',

        # 保存图形
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.transparent': False
    }

    # 使用自定义样式
    with plt.rc_context(custom_style):
        plt.figure()
        plt.plot(x, y1, label='sin(x)')
        plt.plot(x, y2, label='cos(x)')
        plt.title('使用自定义样式表')
        plt.xlabel('X轴')
        plt.ylabel('Y轴')
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\n自定义样式表的关键点:")
    print("1. 可以通过字典定义自定义样式")
    print("2. plt.rc_context()临时应用样式")
    print("3. plt.rcParams全局设置样式")
    print("4. 可以自定义颜色、线条、字体、网格等")
    print("5. cycler()用于定义颜色循环")

def custom_color_maps():
    """
    自定义颜色映射示例
    """
    print("\n" + "=" * 50)
    print("自定义颜色映射")
    print("=" * 50)

    # 创建一些数据用于展示颜色映射
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    # 展示一些内置的颜色映射
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, cmap_name in enumerate(cmaps):
        ax = axes[i]
        im = ax.imshow(Z, cmap=cmap_name, origin='lower')
        ax.set_title(f'颜色映射: {cmap_name}')
        plt.colorbar(im, ax=ax)

    # 创建自定义颜色映射
    colors = [(0, 0, 0.8), (0, 0.8, 0), (0.8, 0, 0)]  # 蓝-绿-红
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    ax = axes[5]
    im = ax.imshow(Z, cmap=custom_cmap, origin='lower')
    ax.set_title('自定义颜色映射')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

    # 展示Seaborn的调色板
    sns_palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, palette in enumerate(sns_palettes):
        ax = axes[i]
        colors = sns.color_palette(palette)
        ax.imshow([colors], aspect='auto')
        ax.set_title(f'Seaborn调色板: {palette}')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    print("\n自定义颜色映射的关键点:")
    print("1. matplotlib.colors.LinearSegmentedColormap: 创建自定义颜色映射")
    print("2. sns.color_palette(): 创建Seaborn调色板")
    print("3. 可以使用RGB元组或十六进制颜色代码定义颜色")
    print("4. 颜色映射适用于热图、等高线图和3D表面")
    print("5. 调色板适用于分类数据")

def consistent_visualization_style():
    """
    创建一致的可视化风格示例
    """
    print("\n" + "=" * 50)
    print("创建一致的可视化风格")
    print("=" * 50)
    
    # 获取样本数据
    data = create_sample_data()
    
    # 定义一个一致的可视化风格
    def apply_consistent_style(ax):
        """应用一致的样式到轴对象"""
        # 设置背景色
        ax.set_facecolor('#f8f8f8')
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        
        # 设置脊柱
        for spine in ax.spines.values():
            spine.set_color('#dddddd')
            spine.set_linewidth(0.8)
        
        # 设置刻度
        ax.tick_params(colors='#555555', direction='out', length=4, width=1)
        
        # 设置标签
        ax.xaxis.label.set_color('#333333')
        ax.yaxis.label.set_color('#333333')
        ax.title.set_color('#333333')
        
        return ax
    
    # 创建一个包含多种图表类型的图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 线图
    ax1 = axes[0, 0]
    ax1.plot(data['line_data']['x'], data['line_data']['y1'], label='sin(x)')
    ax1.plot(data['line_data']['x'], data['line_data']['y2'], label='cos(x)')
    ax1.set_title('线图')
    ax1.set_xlabel('X轴')
    ax1.set_ylabel('Y轴')
    ax1.legend()
    apply_consistent_style(ax1)
    
    # 2. 散点图
    ax2 = axes[0, 1]
    ax2.scatter(data['scatter_data']['x'], data['scatter_data']['y'], alpha=0.7)
    ax2.set_title('散点图')
    ax2.set_xlabel('X轴')
    ax2.set_ylabel('Y轴')
    apply_consistent_style(ax2)
    
    # 3. 条形图
    ax3 = axes[1, 0]
    categories = ['A', 'B', 'C', 'D']
    values = [3, 7, 5, 9]
    ax3.bar(categories, values, color='#5599cc')
    ax3.set_title('条形图')
    ax3.set_xlabel('类别')
    ax3.set_ylabel('值')
    apply_consistent_style(ax3)
    
    # 4. 直方图
    ax4 = axes[1, 1]
    ax4.hist(data['scatter_data']['x'], bins=15, color='#5599cc', alpha=0.7, edgecolor='white')
    ax4.set_title('直方图')
    ax4.set_xlabel('值')
    ax4.set_ylabel('频率')
    apply_consistent_style(ax4)
    
    plt.tight_layout()
    plt.show()
    
    print("\n创建一致的可视化风格的关键点:")
    print("1. 创建一个函数来应用一致的样式")
    print("2. 保持颜色、字体、网格等一致")
    print("3. 可以自定义轴的外观、脊柱和刻度")
    print("4. 一致的样式提高了可视化的专业性")
    print("5. 可以将样式函数应用于任何轴对象")

def publication_ready_figures():
    """
    为出版物准备图形示例
    """
    print("\n" + "=" * 50)
    print("为出版物准备图形")
    print("=" * 50)
    
    # 获取样本数据
    data = create_sample_data()
    
    # 设置出版物质量的样式
    # 在新版本的Matplotlib中，样式名称可能已更改
    try:
        plt.style.use('seaborn-whitegrid')  # 旧版本Matplotlib
    except:
        try:
            plt.style.use('seaborn_whitegrid')  # 新版本Matplotlib
        except:
            # 如果两者都不可用，使用默认样式
            plt.style.use('default')
            print("注意: 'seaborn-whitegrid'样式不可用，使用默认样式代替")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (8, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # 创建图形
    fig, ax = plt.subplots()
    
    # 绘制数据
    x = data['line_data']['x']
    y1 = data['line_data']['y1']
    y2 = data['line_data']['y2']
    
    ax.plot(x, y1, 'o-', label='sin(x)', markersize=4, markevery=10)
    ax.plot(x, y2, 's-', label='cos(x)', markersize=4, markevery=10)
    
    # 添加标题和标签
    ax.set_title('出版物质量图形示例')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    
    # 设置轴范围
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    
    # 添加文本注释
    ax.text(5, 0.8, r'$f(x) = \sin(x)$', fontsize=14)
    ax.text(5, -0.8, r'$g(x) = \cos(x)$', fontsize=14)
    
    # 显示图形
    plt.tight_layout()
    plt.show()
    
    # 恢复默认样式
    plt.style.use('default')
    
    print("\n为出版物准备图形的关键点:")
    print("1. 使用高分辨率(300 DPI或更高)")
    print("2. 选择适合出版物的字体(如Times New Roman)")
    print("3. 确保线条粗细和标记大小适当")
    print("4. 使用LaTeX渲染数学公式")
    print("5. 保存为矢量格式(如PDF或SVG)以便缩放")
    print("6. 考虑图形在黑白打印时的可读性")

def corporate_style_visualization():
    """
    企业风格可视化示例
    """
    print("\n" + "=" * 50)
    print("企业风格可视化")
    print("=" * 50)
    
    # 获取样本数据
    data = create_sample_data()
    
    # 定义企业颜色
    corporate_colors = {
        'primary': '#003366',    # 深蓝色
        'secondary': '#FF9900',  # 橙色
        'tertiary': '#66CC99',   # 绿色
        'quaternary': '#CC3366', # 红色
        'background': '#F5F5F5', # 浅灰色背景
        'text': '#333333',       # 深灰色文本
        'grid': '#DDDDDD'        # 网格线颜色
    }
    
    # 创建企业风格
    corporate_style = {
        'figure.facecolor': corporate_colors['background'],
        'axes.facecolor': 'white',
        'axes.edgecolor': corporate_colors['grid'],
        'axes.labelcolor': corporate_colors['text'],
        'axes.titlecolor': corporate_colors['primary'],
        'axes.grid': True,
        'grid.color': corporate_colors['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'text.color': corporate_colors['text'],
        'xtick.color': corporate_colors['text'],
        'ytick.color': corporate_colors['text'],
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'lines.linewidth': 2.5,
        'patch.edgecolor': 'white',
        'patch.linewidth': 0.5
    }
    
    # 应用企业风格
    with plt.rc_context(corporate_style):
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 线图
        ax1 = axes[0, 0]
        ax1.plot(data['line_data']['x'], data['line_data']['y1'], 
                color=corporate_colors['primary'], label='数据1')
        ax1.plot(data['line_data']['x'], data['line_data']['y2'], 
                color=corporate_colors['secondary'], label='数据2')
        ax1.set_title('企业风格线图')
        ax1.set_xlabel('X轴')
        ax1.set_ylabel('Y轴')
        ax1.legend()
        
        # 2. 条形图
        ax2 = axes[0, 1]
        categories = ['Q1', 'Q2', 'Q3', 'Q4']
        values = [4.5, 6.2, 5.8, 7.4]
        bars = ax2.bar(categories, values, color=corporate_colors['primary'])
        # 突出显示最高值
        bars[3].set_color(corporate_colors['secondary'])
        ax2.set_title('季度业绩')
        ax2.set_xlabel('季度')
        ax2.set_ylabel('收入 (百万)')
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}M', ha='center', va='bottom')
        
        # 3. 饼图
        ax3 = axes[1, 0]
        labels = ['产品A', '产品B', '产品C', '产品D']
        sizes = [35, 25, 20, 20]
        colors = [corporate_colors['primary'], corporate_colors['secondary'], 
                 corporate_colors['tertiary'], corporate_colors['quaternary']]
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        ax3.set_title('产品销售分布')
        ax3.axis('equal')
        
        # 4. 堆叠面积图
        ax4 = axes[1, 1]
        x = np.arange(0, 10, 0.1)
        y1 = np.exp(-x/10) * np.sin(x)
        y2 = np.exp(-x/10) * np.cos(x)
        y3 = np.exp(-x/10) * 0.5
        
        ax4.fill_between(x, 0, y1, alpha=0.7, color=corporate_colors['primary'], label='产品A')
        ax4.fill_between(x, y1, y1+y2, alpha=0.7, color=corporate_colors['secondary'], label='产品B')
        ax4.fill_between(x, y1+y2, y1+y2+y3, alpha=0.7, color=corporate_colors['tertiary'], label='产品C')
        
        ax4.set_title('产品趋势')
        ax4.set_xlabel('时间')
        ax4.set_ylabel('市场份额')
        ax4.legend(loc='upper right')
        
        # 添加企业标志（模拟）
        fig.text(0.02, 0.02, 'COMPANY LOGO', fontsize=14, 
                color=corporate_colors['primary'], weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    print("\n企业风格可视化的关键点:")
    print("1. 使用企业品牌颜色")
    print("2. 保持一致的字体和样式")
    print("3. 添加企业标志")
    print("4. 使用清晰的标题和标签")
    print("5. 突出显示重要数据点")
    print("6. 保持简洁专业的设计")

def run_example():
    """
    运行所有示例
    """
    # Matplotlib内置样式表
    matplotlib_style_sheets()
    
    # 创建自定义样式表
    create_custom_style_sheet()
    
    # 自定义颜色映射
    custom_color_maps()
    
    # 创建一致的可视化风格
    consistent_visualization_style()
    
    # 为出版物准备图形
    publication_ready_figures()
    
    # 企业风格可视化
    corporate_style_visualization()
    
    print("\n" + "=" * 50)
    print("自定义样式和主题教程完成！")
    print("=" * 50)

if __name__ == "__main__":
    run_example()