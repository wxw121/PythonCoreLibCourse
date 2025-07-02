# Python Core Libraries Tutorial

[中文](README.md) | English

This project contains tutorials and example code for Python data science core libraries, including NumPy, Pandas, Matplotlib, Seaborn, PyTorch, and Hugging Face Transformers. Each module provides a complete learning path from basic to advanced levels.

## Project Structure

```
PythonCoreLibCourse/
├── matplotlib_seaborn_tutorial/  # Matplotlib and Seaborn visualization tutorials
├── numpy_tutorial/              # NumPy numerical computing tutorials
├── pandas_tutorial/             # Pandas data analysis tutorials
├── scikit_learn_tutorial/       # Scikit-learn machine learning tutorials
├── pytorch_tutorial/            # PyTorch deep learning tutorials
└── huggingface_transformers/    # Hugging Face Transformers deep learning tutorials
```

## Tutorial Modules

### 1. Matplotlib & Seaborn Tutorial
- Basic plotting functions
- Advanced plotting techniques
- Custom styles and themes
- Seaborn statistical visualization
- Complete Chinese display support

### 2. NumPy Tutorial
- Basic array operations
- Advanced array operations
- Linear algebra computations
- Random number generation
- File input/output

### 3. Pandas Tutorial
- Basic data structures and operations
- Data cleaning and preprocessing
- Data manipulation and transformation
- Data visualization integration

### 4. Scikit-learn Tutorial
- Machine learning basic concepts
- Data preprocessing and feature engineering
- Classification algorithms (Logistic Regression, Decision Trees, Random Forest, SVM, etc.)
- Regression algorithms (Linear Regression, Ridge Regression, Lasso Regression, etc.)
- Clustering algorithms (K-means, DBSCAN, Hierarchical Clustering, etc.)
- Model evaluation and optimization
- Advanced topics (Ensemble Learning, Pipeline Construction, Parameter Optimization, etc.)

### 5. PyTorch Tutorial
- Tensor basic operations
- Automatic differentiation
- Neural network construction
- Convolutional Neural Networks
- Recurrent Neural Networks
- Transfer learning
- Model saving and loading
- GPU acceleration
- Distributed training

### 6. Hugging Face Transformers Tutorial
- Text classification
- Named Entity Recognition
- Question Answering
- Text generation
- Machine translation
- Image classification
- Pre-trained model usage
- Model fine-tuning and training

## Installation and Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PythonCoreLibCourse.git
cd PythonCoreLibCourse
```

2. Install dependencies:
Each tutorial module has its own requirements.txt file, you can install them separately:
```bash
# Install Matplotlib and Seaborn tutorial dependencies
cd matplotlib_seaborn_tutorial
pip install -r requirements.txt

# Install NumPy tutorial dependencies
cd ../numpy_tutorail
pip install -r requirements.txt

# Install Pandas tutorial dependencies
cd ../pandas_tutorial
pip install -r requirements.txt

# Install Scikit-learn tutorial dependencies
cd ../scikit_learn_tutorial
pip install -r requirements.txt

# Install PyTorch tutorial dependencies
cd ../pytorch_tutorial
pip install -r requirements.txt

# Install Hugging Face Transformers tutorial dependencies
cd ../huggingface_transformers
pip install -r requirements.txt
```

3. Run examples:
Each tutorial module has example code that you can run directly:
```bash
# PyTorch examples
python pytorch_tutorial/examples/cnn_mnist.py  # Run CNN image classification example
python pytorch_tutorial/examples/rnn_text.py   # Run RNN text classification example

# Transformers examples
python huggingface_transformers/examples/text_classification.py  # Run text classification example
python huggingface_transformers/examples/translation.py         # Run machine translation example
```

## Usage Instructions

1. Each tutorial module is independent, you can learn selectively based on your needs
2. Example code contains detailed comments and explanations
3. It is recommended to learn in order from basic to advanced
4. You can import modules in Python interactive environment and try modifying parameters to learn
5. For PyTorch and Transformers modules, it is recommended to use GPU for model training and inference acceleration

## Contributing

Issues and Pull Requests are welcome to improve the tutorial content.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
