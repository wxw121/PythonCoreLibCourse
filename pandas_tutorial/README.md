# Pandas 数据分析教程

## 项目概述

这是一个全面的Pandas数据分析教程，旨在帮助初学者和中级用户掌握Pandas库的基本用法和高级功能。通过实际示例和详细解释，本教程涵盖了数据操作、清洗、分析和可视化等方面的内容。

## 安装说明

### 前提条件

- Python 3.6+
- pip (Python包管理器)

### 安装依赖

在使用本教程之前，请确保安装了所需的依赖库：

```bash
pip install pandas numpy matplotlib seaborn
```

## 使用方法

1. 克隆或下载本仓库
2. 进入项目目录
3. 运行主程序：

```bash
python main.py
```

4. 在菜单中选择要学习的模块

## 教程内容

本教程包含以下模块：

### 1. Pandas基础知识 (`pandas_basics.py`)

- Series基础知识
- DataFrame基础知识
- 索引和选择操作
  - 标签索引 (.loc)
  - 位置索引 (.iloc)
  - 布尔索引
  - 混合索引

### 2. Pandas数据操作和转换 (`pandas_data_manipulation.py`)

- 合并和连接操作
  - concat方法
  - merge方法
  - join方法
- 数据重塑操作
  - pivot方法
  - pivot_table方法
  - stack和unstack方法
  - melt方法
- 分组操作
  - 基本分组
  - 聚合操作
  - 转换操作
  - 过滤操作
  - 应用操作
- 时间序列操作
  - 创建时间序列
  - 时间序列索引
  - 重采样
  - 移动窗口函数
  - 时区处理

### 3. Pandas数据清洗和预处理 (`pandas_data_cleaning.py`)

- 处理缺失值
  - 检测缺失值
  - 删除缺失值
  - 填充缺失值
  - 插值
  - 替换特定值
- 处理重复值
  - 检测重复值
  - 删除重复值
  - 计数和汇总重复值
- 处理异常值
  - 使用统计方法检测异常值
  - 过滤异常值
  - 替换异常值
  - 可视化异常值
- 数据转换
  - 类型转换
  - 标准化和归一化
  - 编码分类变量
  - 字符串操作

### 4. Pandas数据可视化 (`pandas_visualization.py`)

- 基本绘图
  - 线图
  - 柱状图
  - 散点图
  - 直方图
  - 箱线图
- 统计图表
  - 密度图
  - 相关性热图
  - 成对关系图
  - 小提琴图
  - KDE图
- 多子图示例
  - 基本子图布局
  - 不同大小的子图
  - 组合图表

## 项目结构

```
pandas_tutorial/
├── README.md                    # 项目说明文档
├── main.py                      # 主入口程序
├── pandas_basics.py             # Pandas基础知识
├── pandas_data_manipulation.py  # 数据操作和转换
├── pandas_data_cleaning.py      # 数据清洗和预处理
└── pandas_visualization.py      # 数据可视化
```

## 学习建议

1. 按照模块顺序学习，从基础知识开始
2. 尝试修改示例代码，观察结果变化
3. 将所学知识应用到自己的数据集上
4. 参考Pandas官方文档深入学习：[Pandas Documentation](https://pandas.pydata.org/docs/)

## 贡献

欢迎提出建议和改进意见！如果您发现任何错误或有改进建议，请提交issue或pull request。

## 许可

本项目采用MIT许可证。详见LICENSE文件。
