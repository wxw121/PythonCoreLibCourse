"""
NumPy教程项目
本项目提供了NumPy库的全面教程，包括基础知识、高级功能、线性代数、随机数生成和数据输入输出等内容。
作者：Craft
"""

import numpy as np
import sys

def print_header(title):
    """打印格式化的标题"""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60 + "\n")

def introduction():
    """介绍NumPy教程项目"""
    print_header("NumPy教程项目介绍")
    
    print("欢迎使用NumPy教程项目！")
    print("\nNumPy是Python中用于科学计算的基础库，提供了多维数组对象、各种派生对象")
    print("以及用于数组快速运算的各种函数。NumPy是许多其他科学计算库的基础，")
    print("如SciPy、Pandas、Matplotlib等。")
    
    print("\n本教程项目包含以下模块：")
    print("1. numpy_basics.py - NumPy基础知识")
    print("2. numpy_advanced.py - NumPy高级功能")
    print("3. numpy_linear_algebra.py - NumPy线性代数")
    print("4. numpy_random.py - NumPy随机数生成")
    print("5. numpy_io.py - NumPy数据输入输出")
    
    print("\n每个模块都包含详细的文字解释和代码示例，可以单独运行。")
    print("建议按照上述顺序学习，从基础到高级逐步深入。")

def check_numpy_installation():
    """检查NumPy是否已安装，并显示版本信息"""
    print_header("NumPy安装检查")
    
    try:
        print(f"NumPy版本: {np.__version__}")
        print(f"NumPy安装路径: {np.__path__}")
        print("\nNumPy已正确安装！")
    except:
        print("NumPy未安装或无法导入。")
        print("请使用以下命令安装NumPy:")
        print("pip install numpy")
        sys.exit(1)

def quick_examples():
    """展示一些NumPy的快速示例"""
    print_header("NumPy快速示例")
    
    # 创建数组
    print("1. 创建数组:")
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    
    print(f"一维数组: {arr1}")
    print(f"二维数组:\n{arr2}")
    
    # 数组操作
    print("\n2. 数组操作:")
    print(f"数组形状: {arr2.shape}")
    print(f"数组维度: {arr2.ndim}")
    print(f"数组元素类型: {arr2.dtype}")
    print(f"数组元素总数: {arr2.size}")
    
    # 数组索引和切片
    print("\n3. 数组索引和切片:")
    print(f"arr2[0, 1] = {arr2[0, 1]}")  # 第一行第二列的元素
    print(f"arr2[0] = {arr2[0]}")        # 第一行
    print(f"arr2[:, 1] = {arr2[:, 1]}")  # 第二列
    
    # 数组运算
    print("\n4. 数组运算:")
    print(f"arr1 + 10 = {arr1 + 10}")
    print(f"arr1 * 2 = {arr1 * 2}")
    print(f"arr1 的平均值: {arr1.mean()}")
    print(f"arr1 的总和: {arr1.sum()}")
    print(f"arr1 的最小值: {arr1.min()}")
    print(f"arr1 的最大值: {arr1.max()}")
    
    # 数组变形
    print("\n5. 数组变形:")
    arr3 = np.arange(12)
    print(f"原始数组: {arr3}")
    reshaped = arr3.reshape(3, 4)
    print(f"重塑为3x4数组:\n{reshaped}")
    
    # 数组合并和分割
    print("\n6. 数组合并和分割:")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    print(f"水平合并:\n{np.hstack((a, b))}")
    print(f"垂直合并:\n{np.vstack((a, b))}")

def run_module(module_name):
    """运行指定的教程模块"""
    try:
        if module_name == "basics":
            import numpy_basics
            numpy_basics.main()
        elif module_name == "advanced":
            import numpy_advanced
            numpy_advanced.main()
        elif module_name == "linalg":
            import numpy_linear_algebra
            numpy_linear_algebra.main()
        elif module_name == "random":
            import numpy_random
            numpy_random.main()
        elif module_name == "io":
            import numpy_io
            # 假设numpy_io模块有main函数
            if hasattr(numpy_io, 'main'):
                numpy_io.main()
            else:
                print("numpy_io模块没有main函数，请直接导入并使用其中的函数。")
        else:
            print(f"未知模块: {module_name}")
            print("可用模块: basics, advanced, linalg, random, io")
    except ImportError as e:
        print(f"无法导入模块: {e}")
    except Exception as e:
        print(f"运行模块时出错: {e}")

def show_usage():
    """显示使用说明"""
    print_header("使用说明")
    
    print("运行整个教程:")
    print("python main.py")
    print("\n运行特定模块:")
    print("python main.py <module_name>")
    print("\n可用模块:")
    print("- basics   : NumPy基础知识")
    print("- advanced : NumPy高级功能")
    print("- linalg   : NumPy线性代数")
    print("- random   : NumPy随机数生成")
    print("- io       : NumPy数据输入输出")

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ["help", "-h", "--help"]:
            show_usage()
        else:
            run_module(sys.argv[1])
    else:
        # 运行完整教程
        introduction()
        check_numpy_installation()
        quick_examples()
        show_usage()

if __name__ == "__main__":
    main()
