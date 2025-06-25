"""
Matplotlib和Seaborn教程包

这个包提供了一系列关于Matplotlib和Seaborn数据可视化的教程示例。
每个模块都包含了详细的示例代码和说明，帮助用户学习和掌握这两个强大的可视化库。

模块列表：
1. matplotlib_basics - Matplotlib基础知识
   - 基本图表类型
   - 子图和布局
   - 基本定制化

2. matplotlib_advanced - Matplotlib高级特性
   - 3D绘图
   - 动画
   - 自定义投影
   - 图像处理
   - 高级样式设置

3. seaborn_basics - Seaborn基础知识
   - 统计图表
   - 分布图
   - 关系图
   - 分类图

4. seaborn_advanced - Seaborn高级特性
   - 复杂统计可视化
   - 多变量分析
   - 回归分析图
   - 高级分类数据可视化
   - 自定义调色板和样式

5. custom_styles - 自定义样式和主题
   - 创建自定义样式表
   - 自定义颜色映射和调色板
   - 创建一致的可视化风格
   - 为出版物准备图形
   - 创建企业风格的可视化

使用说明：
1. 每个模块都可以独立运行，包含完整的示例
2. 建议按照基础到高级的顺序学习
3. 所有示例都包含详细的注释和说明
4. 可以通过修改示例代码来实验不同的可视化效果

依赖要求：
- Python 3.6+
- matplotlib
- seaborn
- numpy
- pandas

示例：
>>> from matplotlib_seaborn_tutorial import matplotlib_basics
>>> matplotlib_basics.run_example()

>>> from matplotlib_seaborn_tutorial import seaborn_basics
>>> seaborn_basics.run_example()
"""

# 版本信息
__version__ = '1.0.0'

# 导入所有模块
from . import config
from . import matplotlib_basics
from . import matplotlib_advanced
from . import seaborn_basics
from . import seaborn_advanced
from . import custom_styles

# 定义包的所有模块
__all__ = [
    'config',
    'matplotlib_basics',
    'matplotlib_advanced',
    'seaborn_basics',
    'seaborn_advanced',
    'custom_styles'
]
