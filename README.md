# Python核心库教程

这个项目包含了Python数据科学核心库的教程和示例代码，包括NumPy、Pandas、Matplotlib、Seaborn、PyTorch以及Hugging Face Transformers。每个模块都提供了从基础到高级的完整学习路径。

## 项目结构

```
PythonCoreLibCourse/
├── matplotlib_seaborn_tutorial/  # Matplotlib和Seaborn数据可视化教程
├── numpy_tutorial/              # NumPy数值计算教程
├── pandas_tutorial/             # Pandas数据分析教程
├── scikit_learn_tutorial/       # Scikit-learn机器学习教程
├── pytorch_tutorial/            # PyTorch深度学习教程
└── huggingface_transformers/    # Hugging Face Transformers深度学习教程
```

## 教程模块

### 1. Matplotlib & Seaborn教程
- 基础绘图功能
- 高级绘图技巧
- 自定义样式和主题
- Seaborn统计可视化
- 完整的中文显示支持

### 2. NumPy教程
- 数组基础操作
- 高级数组操作
- 线性代数计算
- 随机数生成
- 文件输入输出

### 3. Pandas教程
- 基础数据结构和操作
- 数据清洗和预处理
- 数据操作和转换
- 数据可视化集成

### 4. Scikit-learn教程
- 机器学习基础概念
- 数据预处理和特征工程
- 分类算法（逻辑回归、决策树、随机森林、SVM等）
- 回归算法（线性回归、岭回归、Lasso回归等）
- 聚类算法（K-means、DBSCAN、层次聚类等）
- 模型评估与优化
- 高级主题（集成学习、管道构建、参数优化等）

### 5. PyTorch教程
- 张量基础操作
- 自动微分
- 神经网络构建
- 卷积神经网络
- 循环神经网络
- 迁移学习
- 模型保存与加载
- GPU加速计算
- 分布式训练

### 6. Hugging Face Transformers教程
- 文本分类
- 命名实体识别
- 问答系统
- 文本生成
- 机器翻译
- 图像分类
- 预训练模型使用
- 模型微调和训练

## 安装和使用

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/PythonCoreLibCourse.git
cd PythonCoreLibCourse
```

2. 安装依赖：
每个教程模块都有自己的requirements.txt文件，你可以分别安装：
```bash
# 安装Matplotlib和Seaborn教程依赖
cd matplotlib_seaborn_tutorial
pip install -r requirements.txt

# 安装NumPy教程依赖
cd ../numpy_tutorail
pip install -r requirements.txt

# 安装Pandas教程依赖
cd ../pandas_tutorial
pip install -r requirements.txt

# 安装Scikit-learn教程依赖
cd ../scikit_learn_tutorial
pip install -r requirements.txt

# 安装PyTorch教程依赖
cd ../pytorch_tutorial
pip install -r requirements.txt

# 安装Hugging Face Transformers教程依赖
cd ../huggingface_transformers
pip install -r requirements.txt
```

3. 运行示例：
每个教程模块都有示例代码，可以直接运行查看：
```bash
# PyTorch示例
python pytorch_tutorial/examples/cnn_mnist.py  # 运行CNN图像分类示例
python pytorch_tutorial/examples/rnn_text.py   # 运行RNN文本分类示例

# Transformers示例
python huggingface_transformers/examples/text_classification.py  # 运行文本分类示例
python huggingface_transformers/examples/translation.py         # 运行机器翻译示例
```

## 使用说明

1. 每个教程模块都是独立的，你可以根据需要选择性学习
2. 示例代码中包含详细的注释和说明
3. 建议按照基础到高级的顺序学习
4. 可以在Python交互式环境中导入模块，尝试修改参数来学习
5. 对于PyTorch和Transformers模块，建议使用GPU来加速模型训练和推理

## 贡献

欢迎提交Issue和Pull Request来完善教程内容。

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。
