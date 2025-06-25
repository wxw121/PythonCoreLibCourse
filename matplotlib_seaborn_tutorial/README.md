# Matplotlib 和 Seaborn 教程

这个教程包提供了一系列关于 Matplotlib 和 Seaborn 数据可视化的详细示例和说明。通过这些教程，你可以学习如何使用这两个强大的 Python 可视化库创建各种类型的图表和可视化效果。

## 目录结构

```
matplotlib_seaborn_tutorial/
├── __init__.py              # 包初始化文件
├── config.py                # 包配置文件
├── matplotlib_basics.py     # Matplotlib基础知识
├── matplotlib_advanced.py   # Matplotlib高级特性
├── seaborn_basics.py       # Seaborn基础知识
├── seaborn_advanced.py     # Seaborn高级特性
├── custom_styles.py        # 自定义样式和主题
└── README.md               # 本文档
```

## 安装要求

- Python 3.6+
- matplotlib
- seaborn
- numpy
- pandas

可以使用以下命令安装所需的依赖：

```bash
pip install matplotlib seaborn numpy pandas
```

## 模块说明

### 1. Matplotlib基础知识 (matplotlib_basics.py)
- 基本图表类型（线图、散点图、条形图等）
- 子图和布局
- 坐标轴设置
- 图例和标签
- 基本样式定制

### 2. Matplotlib高级特性 (matplotlib_advanced.py)
- 3D绘图
- 动画效果
- 自定义投影
- 图像处理
- 高级样式设置

### 3. Seaborn基础知识 (seaborn_basics.py)
- 统计图表
- 分布图
- 关系图
- 分类图
- 基本样式设置

### 4. Seaborn高级特性 (seaborn_advanced.py)
- 复杂统计可视化
- 多变量分析
- 回归分析图
- 高级分类数据可视化
- 自定义调色板

### 5. 自定义样式和主题 (custom_styles.py)
- 创建自定义样式表
- 自定义颜色映射
- 创建一致的可视化风格
- 为出版物准备图形
- 企业风格可视化

## 使用方法

### 1. 运行单个模块示例

```python
# 运行Matplotlib基础教程
from matplotlib_seaborn_tutorial import matplotlib_basics
matplotlib_basics.run_example()

# 运行Seaborn基础教程
from matplotlib_seaborn_tutorial import seaborn_basics
seaborn_basics.run_example()
```

### 2. 使用特定功能

```python
# 使用Matplotlib绘制基本图表
from matplotlib_seaborn_tutorial.matplotlib_basics import plot_basic_charts
plot_basic_charts()

# 使用Seaborn创建统计图表
from matplotlib_seaborn_tutorial.seaborn_basics import distribution_plots
distribution_plots(data)
```

### 3. 自定义样式

```python
# 使用自定义样式
from matplotlib_seaborn_tutorial.custom_styles import create_custom_style_sheet
create_custom_style_sheet()
```

## 学习建议

1. 按照以下顺序学习各个模块：
   - 首先学习 matplotlib_basics.py
   - 然后是 seaborn_basics.py
   - 接着是 matplotlib_advanced.py 和 seaborn_advanced.py
   - 最后学习 custom_styles.py

2. 每个示例都包含详细的注释和说明，建议仔细阅读并理解代码

3. 尝试修改示例代码中的参数，观察不同的可视化效果

4. 将学到的知识应用到自己的数据可视化项目中

## 注意事项

- 所有示例都包含完整的代码和注释
- 示例中使用的数据都是随机生成的，可以替换为自己的数据
- 部分高级功能可能需要额外的依赖包
- 建议在Jupyter Notebook或类似的交互环境中运行示例

## 贡献

欢迎提供改进建议和贡献代码。可以通过以下方式参与：

1. 提交Issue报告问题或建议
2. 提交Pull Request贡献代码
3. 完善文档和示例

## 许可证

本教程采用 MIT 许可证。详见 LICENSE 文件。
