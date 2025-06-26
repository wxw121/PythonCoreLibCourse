"""
Matplotlib配置模块

此模块包含matplotlib的基本配置，包括：
1. 中文字体支持
2. 图表样式设置
3. 默认参数配置
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import sys

# 设置全局编码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 黑体、微软雅黑、宋体
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 设置全局字体
mpl.rcParams['font.family'] = plt.rcParams['font.sans-serif'][0]

# 设置DPI
plt.rcParams['figure.dpi'] = 100

# 设置默认颜色循环
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
])

# 设置默认图表大小
plt.rcParams['figure.figsize'] = [10, 6]

# 设置网格样式
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

# 设置默认保存格式
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# 设置轴标签和标题的字体大小
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# 设置图例字体大小和位置
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.loc'] = 'best'

# 设置刻度标签大小
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
