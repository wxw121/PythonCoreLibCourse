"""
NumPy教程
本教程介绍NumPy库的高级功能，包括广播、通用函数、结构化数组等内容。
"""

import numpy as np

def broadcasting():
    """
    演示NumPy的广播机制
    广播是NumPy在算术运算期间处理不同形状数组的方式
    """
    print("="*50)
    print("NumPy广播机制")
    print("="*50)
    
    # 创建一个2D数组
    arr_2d = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    
    # 创建一个1D数组
    arr_1d = np.array([10, 20, 30])
    
    print("\n1. 原始数组:")
    print(f"2D数组:\n{arr_2d}")
    print(f"1D数组: {arr_1d}")
    
    # 使用广播进行加法运算
    # 1D数组会被"广播"到与2D数组相同的形状
    result = arr_2d + arr_1d
    
    print("\n2. 广播加法结果:")
    print(result)
    print("\n解释: 1D数组[10,20,30]被广播到每一行，相当于:")
    print("[[1,2,3] + [10,20,30],")
    print(" [4,5,6] + [10,20,30],")
    print(" [7,8,9] + [10,20,30]]")
    
    # 使用标量进行广播
    scalar = 10
    result_scalar = arr_2d * scalar
    
    print("\n3. 标量广播结果:")
    print(result_scalar)
    print("\n解释: 标量10被广播到整个数组，每个元素都乘以10")

def universal_functions():
    """
    演示NumPy的通用函数(ufuncs)
    通用函数是对数组中每个元素进行操作的函数
    """
    print("\n"+"="*50)
    print("NumPy通用函数")
    print("="*50)
    
    # 创建示例数组
    arr = np.array([0, np.pi/4, np.pi/2, np.pi])
    
    print("\n1. 原始数组:")
    print(arr)
    
    # 应用三角函数
    print("\n2. 三角函数:")
    print(f"sin(arr) = {np.sin(arr)}")
    print(f"cos(arr) = {np.cos(arr)}")
    print(f"tan(arr) = {np.tan(arr)}")
    
    # 指数和对数函数
    arr2 = np.array([1, 2, 3, 4])
    print("\n3. 指数和对数函数:")
    print(f"原始数组: {arr2}")
    print(f"exp(arr2) = {np.exp(arr2)}")
    print(f"log(arr2) = {np.log(arr2)}")
    print(f"sqrt(arr2) = {np.sqrt(arr2)}")

    # 舍入函数
    arr3 = np.array([1.2, 2.7, 3.5, 4.9])
    print("\n4. 舍入函数:")
    print(f"原始数组: {arr3}")
    print(f"向下取整 floor(arr3) = {np.floor(arr3)}")
    print(f"向上取整 ceil(arr3) = {np.ceil(arr3)}")
    print(f"四舍五入 round(arr3) = {np.round(arr3)}")

def structured_arrays():
    """
    演示NumPy的结构化数组
    结构化数组是具有复合数据类型的数组
    """
    print("\n"+"="*50)
    print("NumPy结构化数组")
    print("="*50)

    # 定义结构化数据类型
    dt = np.dtype([('name', 'U20'), ('age', 'i4'), ('salary', 'f8')])

    # 创建结构化数组
    employees = np.array([
        ('John', 25, 60000.0),
        ('Jane', 30, 75000.0),
        ('Bob', 35, 80000.0)
    ], dtype=dt)

    print("\n1. 结构化数组:")
    print(employees)

    # 访问特定字段
    print("\n2. 访问特定字段:")
    print(f"所有名字: {employees['name']}")
    print(f"所有年龄: {employees['age']}")
    print(f"所有薪资: {employees['salary']}")

    # 访问特定记录
    print("\n3. 访问特定记录:")
    print(f"第一个员工: {employees[0]}")
    print(f"第二个员工的名字: {employees[1]['name']}")

    # 基于字段进行排序
    print("\n4. 基于年龄排序:")
    sorted_by_age = np.sort(employees, order='age')
    print(sorted_by_age)

def array_manipulation():
    """
    演示NumPy的数组操作函数
    包括：连接、分割、添加/删除元素等
    """
    print("\n"+"="*50)
    print("NumPy数组操作")
    print("="*50)

    # 创建示例数组
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])

    print("\n1. 原始数组:")
    print(f"数组1:\n{arr1}")
    print(f"数组2:\n{arr2}")

    # 连接数组
    print("\n2. 连接数组:")
    # 水平连接（按列）
    h_concat = np.hstack((arr1, arr2))
    print("水平连接 (hstack):")
    print(h_concat)

    # 垂直连接（按行）
    v_concat = np.vstack((arr1, arr2))
    print("\n垂直连接 (vstack):")
    print(v_concat)

    # 深度连接（按深度/第三维）
    d_concat = np.dstack((arr1, arr2))
    print("\n深度连接 (dstack):")
    print(d_concat)

    # 分割数组
    arr3 = np.arange(16).reshape(4, 4)
    print("\n3. 分割数组:")
    print(f"原始数组:\n{arr3}")

    # 水平分割（按列）
    h_split = np.hsplit(arr3, 2)
    print("\n水平分割 (hsplit):")
    print(f"第一部分:\n{h_split[0]}")
    print(f"第二部分:\n{h_split[1]}")

    # 垂直分割（按行）
    v_split = np.vsplit(arr3, 2)
    print("\n垂直分割 (vsplit):")
    print(f"第一部分:\n{v_split[0]}")
    print(f"第二部分:\n{v_split[1]}")

def main():
    """
    主函数，按顺序运行所有示例
    """
    broadcasting()
    universal_functions()
    structured_arrays()
    array_manipulation()

if __name__ == "__main__":
    main()
