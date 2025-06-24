"""
NumPy基础教程
本教程介绍NumPy库的基础知识，包括数组创建、基本操作、索引和切片等内容。
"""

import numpy as np

def numpy_introduction():
    """
    NumPy简介和基本概念
    NumPy是Python中用于数值计算的基础库，它提供了：
    1. 多维数组对象
    2. 各种派生对象（如掩码数组和矩阵）
    3. 用于数组快速运算的各种函数
    4. 用于数组读写的工具
    5. 线性代数、傅里叶变换和随机数生成等功能
    """
    print("="*50)
    print("NumPy简介")
    print("="*50)

    # 创建一个简单的数组
    arr = np.array([1, 2, 3, 4, 5])
    print("\n1. 创建的简单数组:")
    print(arr)

    # 显示数组的基本属性
    print("\n2. 数组的基本属性:")
    print(f"数组的维度（shape）: {arr.shape}")
    print(f"数组的数据类型（dtype）: {arr.dtype}")
    print(f"数组的维数（ndim）: {arr.ndim}")

def array_creation():
    """
    演示不同的数组创建方法
    包括：从列表创建、使用NumPy函数创建、创建特殊数组等
    """
    print("\n"+"="*50)
    print("数组创建方法")
    print("="*50)

    # 1. 从列表创建数组
    list_arr = np.array([[1, 2, 3], [4, 5, 6]])
    print("\n1. 从列表创建2D数组:")
    print(list_arr)

    # 2. 创建全零数组
    zeros_arr = np.zeros((3, 4))  # 创建3行4列的全零数组
    print("\n2. 创建全零数组:")
    print(zeros_arr)

    # 3. 创建全一数组
    ones_arr = np.ones((2, 3))    # 创建2行3列的全一数组
    print("\n3. 创建全一数组:")
    print(ones_arr)

    # 4. 创建等差数列
    arange_arr = np.arange(0, 10, 2)  # 从0到10，步长为2
    print("\n4. 创建等差数列:")
    print(arange_arr)

    # 5. 创建线性等分数组
    linspace_arr = np.linspace(0, 1, 5)  # 在0和1之间创建5个等距点
    print("\n5. 创建线性等分数组:")
    print(linspace_arr)

def array_indexing():
    """
    演示NumPy数组的索引和切片操作
    包括：基本索引、切片、布尔索引等
    """
    print("\n"+"="*50)
    print("数组索引和切片")
    print("="*50)

    # 创建一个2D数组用于演示
    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])
    print("\n示例数组:")
    print(arr)

    # 1. 基本索引
    print("\n1. 基本索引:")
    print(f"arr[0, 0] = {arr[0, 0]}")  # 第一行第一列的元素
    print(f"arr[2, 3] = {arr[2, 3]}")  # 最后一行最后一列的元素

    # 2. 切片操作
    print("\n2. 切片操作:")
    print("前两行:")
    print(arr[:2])
    print("\n所有行的前三列:")
    print(arr[:, :3])

    # 3. 布尔索引
    print("\n3. 布尔索引:")
    bool_mask = arr > 6  # 创建布尔掩码
    print("大于6的元素:")
    print(arr[bool_mask])

def basic_operations():
    """
    演示NumPy数组的基本运算操作
    包括：算术运算、统计运算、数组变形等
    """
    print("\n"+"="*50)
    print("基本运算操作")
    print("="*50)

    # 创建示例数组
    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([5, 6, 7, 8])

    # 1. 算术运算
    print("\n1. 算术运算:")
    print(f"数组1: {arr1}")
    print(f"数组2: {arr2}")
    print(f"加法: {arr1 + arr2}")
    print(f"乘法: {arr1 * arr2}")
    print(f"数组1的平方: {arr1 ** 2}")

    # 2. 统计运算
    arr3 = np.array([[1, 2, 3], [4, 5, 6]])
    print("\n2. 统计运算:")
    print(f"数组:\n{arr3}")
    print(f"平均值: {arr3.mean()}")
    print(f"最大值: {arr3.max()}")
    print(f"最小值: {arr3.min()}")
    print(f"行的和: {arr3.sum(axis=1)}")
    print(f"列的和: {arr3.sum(axis=0)}")

    # 3. 数组变形
    print("\n3. 数组变形:")
    arr4 = np.arange(12)
    print(f"原始数组: {arr4}")
    reshaped_arr = arr4.reshape(3, 4)
    print("重塑为3x4数组:")
    print(reshaped_arr)

def main():
    """
    主函数，按顺序运行所有示例
    """
    numpy_introduction()
    array_creation()
    array_indexing()
    basic_operations()

if __name__ == "__main__":
    main()
