"""
NumPy数据输入输出教程
本教程介绍NumPy库的数据输入输出功能，包括数组的保存和加载、文本文件和二进制文件的读写等内容。
"""

import numpy as np
import os
import tempfile

def save_load_arrays():
    """
    演示如何保存和加载NumPy数组
    包括：npy格式、npz格式、文本格式
    """
    print("="*50)
    print("NumPy数组的保存和加载")
    print("="*50)

    # 创建示例数组
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])

    print("\n1. 创建的数组:")
    print(f"arr1: {arr1}")
    print(f"arr2:\n{arr2}")

    # 创建临时目录用于保存文件
    temp_dir = tempfile.mkdtemp()
    print(f"\n临时目录: {temp_dir}")

    # 1. 使用.npy格式保存单个数组
    print("\n2. 使用.npy格式保存单个数组:")
    npy_file = os.path.join(temp_dir, 'array.npy')
    np.save(npy_file, arr1)
    print(f"数组已保存到: {npy_file}")

    # 加载.npy文件
    loaded_arr = np.load(npy_file)
    print(f"从.npy文件加载的数组: {loaded_arr}")
    print(f"原始数组和加载的数组是否相同: {np.array_equal(arr1, loaded_arr)}")

    # 2. 使用.npz格式保存多个数组
    print("\n3. 使用.npz格式保存多个数组:")
    npz_file = os.path.join(temp_dir, 'arrays.npz')
    np.savez(npz_file, array1=arr1, array2=arr2)
    print(f"多个数组已保存到: {npz_file}")

    # 加载.npz文件
    loaded_arrays = np.load(npz_file)
    print(f"从.npz文件加载的数组: {list(loaded_arrays.keys())}")
    print(f"array1: {loaded_arrays['array1']}")
    print(f"array2:\n{loaded_arrays['array2']}")

    # 3. 使用压缩的.npz格式保存多个数组
    print("\n4. 使用压缩的.npz格式保存多个数组:")
    npz_compressed_file = os.path.join(temp_dir, 'arrays_compressed.npz')
    np.savez_compressed(npz_compressed_file, array1=arr1, array2=arr2)
    print(f"压缩的多个数组已保存到: {npz_compressed_file}")

    # 加载压缩的.npz文件
    loaded_compressed = np.load(npz_compressed_file)
    print(f"从压缩的.npz文件加载的数组: {list(loaded_compressed.keys())}")

    # 4. 使用文本格式保存数组
    print("\n5. 使用文本格式保存数组:")
    txt_file = os.path.join(temp_dir, 'array.txt')
    np.savetxt(txt_file, arr2, fmt='%d', delimiter=',', header='col1,col2,col3')
    print(f"数组已保存为文本文件: {txt_file}")

    # 显示文本文件内容
    with open(txt_file, 'r') as f:
        print("\n文本文件内容:")
        print(f.read())

    # 从文本文件加载数组
    loaded_txt = np.loadtxt(txt_file, delimiter=',', skiprows=1)
    print(f"从文本文件加载的数组:\n{loaded_txt}")

def text_file_io():
    """
    演示NumPy的文本文件输入输出功能
    包括：loadtxt, savetxt, genfromtxt
    """
    print("\n"+"="*50)
    print("NumPy文本文件输入输出")
    print("="*50)

    # 创建临时目录用于保存文件
    temp_dir = tempfile.mkdtemp()

    # 1. 创建示例数据
    print("\n1. 创建示例数据:")
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    print(f"示例数据:\n{data}")

    # 2. 使用savetxt保存数据
    print("\n2. 使用savetxt保存数据:")
    txt_file = os.path.join(temp_dir, 'data.csv')
    np.savetxt(
        txt_file,
        data,
        delimiter=',',
        fmt='%.2f',
        header='col1,col2,col3',
        comments='# '
    )
    print(f"数据已保存到: {txt_file}")

    # 显示文件内容
    with open(txt_file, 'r') as f:
        print("\n文件内容:")
        print(f.read())

    # 3. 使用loadtxt加载数据
    print("\n3. 使用loadtxt加载数据:")
    loaded_data = np.loadtxt(txt_file, delimiter=',', skiprows=1)
    print(f"加载的数据:\n{loaded_data}")

    # 4. 创建包含缺失值的数据
    print("\n4. 处理包含缺失值的数据:")
    missing_data_file = os.path.join(temp_dir, 'missing_data.csv')
    with open(missing_data_file, 'w') as f:
        f.write("# col1,col2,col3\n")
        f.write("1.0,2.0,3.0\n")
        f.write("4.0,,6.0\n")  # 缺失值
        f.write("7.0,8.0,9.0\n")

    print(f"已创建包含缺失值的文件: {missing_data_file}")

    # 显示文件内容
    with open(missing_data_file, 'r') as f:
        print("\n包含缺失值的文件内容:")
        print(f.read())

    # 5. 使用genfromtxt处理缺失值
    print("\n5. 使用genfromtxt处理缺失值:")
    missing_data = np.genfromtxt(
        missing_data_file,
        delimiter=',',
        skip_header=1,
        filling_values=-999  # 用-999替换缺失值
    )
    print(f"处理缺失值后的数据:\n{missing_data}")

def binary_file_io():
    """
    演示NumPy的二进制文件输入输出功能
    包括：tofile, fromfile
    """
    print("\n"+"="*50)
    print("NumPy二进制文件输入输出")
    print("="*50)

    # 创建临时目录用于保存文件
    temp_dir = tempfile.mkdtemp()

    # 1. 创建示例数组
    print("\n1. 创建示例数组:")
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    print(f"示例数组:\n{arr}")
    print(f"数组形状: {arr.shape}")
    print(f"数据类型: {arr.dtype}")

    # 2. 将数组保存为二进制文件
    print("\n2. 将数组保存为二进制文件:")
    binary_file = os.path.join(temp_dir, 'array.bin')
    arr.tofile(binary_file)
    print(f"数组已保存到二进制文件: {binary_file}")

    # 3. 从二进制文件加载数组
    print("\n3. 从二进制文件加载数组:")
    # 注意：fromfile需要知道数据类型和形状
    loaded_arr = np.fromfile(binary_file, dtype=np.int32)
    print(f"加载的一维数组: {loaded_arr}")

    # 重塑为原始形状
    loaded_arr = loaded_arr.reshape(2, 3)
    print(f"重塑后的数组:\n{loaded_arr}")

    # 4. 使用memmap进行内存映射
    print("\n4. 使用memmap进行内存映射:")
    memmap_file = os.path.join(temp_dir, 'memmap.dat')

    # 创建内存映射文件
    mm = np.memmap(memmap_file, dtype=np.float32, mode='w+', shape=(3, 4))
    print(f"创建的内存映射数组:\n{mm}")

    # 修改内存映射数组
    mm[:] = np.arange(12).reshape(3, 4)
    print(f"修改后的内存映射数组:\n{mm}")

    # 将更改刷新到磁盘
    mm.flush()

    # 重新打开内存映射文件
    mm_readonly = np.memmap(memmap_file, dtype=np.float32, mode='r', shape=(3, 4))
    print(f"重新打开的内存映射数组:\n{mm_readonly}")

    print("\n内存映射的优点:")
    print("1. 可以处理大于内存的数组")
    print("2. 多个进程可以共享同一个内存映射文件")
    print("3. 更改会直接写入磁盘，无需额外的保存步骤")

def interoperability():
    """
    演示NumPy与其他格式的互操作性
    包括：CSV、Excel、图像等
    """
    print("\n"+"="*50)
    print("NumPy与其他格式的互操作性")
    print("="*50)

    # 创建临时目录用于保存文件
    temp_dir = tempfile.mkdtemp()

    # 1. CSV文件
    print("\n1. CSV文件:")
    # 创建示例数据
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    # 保存为CSV
    csv_file = os.path.join(temp_dir, 'data.csv')
    np.savetxt(csv_file, data, delimiter=',', fmt='%d')
    print(f"数据已保存为CSV: {csv_file}")

    # 使用Python标准库读取CSV
    print("\n使用Python标准库读取CSV:")
    import csv
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            print(row)

    # 2. 与Pandas的互操作性
    print("\n2. 与Pandas的互操作性:")
    print("注意: 以下代码需要安装pandas库才能运行")
    print("示例代码:")
    print("```python")
    print("import pandas as pd")
    print("# NumPy数组转换为Pandas DataFrame")
    print("df = pd.DataFrame(data, columns=['A', 'B', 'C'])")
    print("print(df)")
    print("")
    print("# Pandas DataFrame转换为NumPy数组")
    print("arr = df.to_numpy()")
    print("print(arr)")
    print("```")

    # 3. 图像处理
    print("\n3. 图像处理:")
    print("注意: 以下代码需要安装Pillow库才能运行")
    print("示例代码:")
    print("```python")
    print("from PIL import Image")
    print("# 创建一个简单的灰度图像")
    print("img_array = np.linspace(0, 255, 100*100).reshape(100, 100).astype(np.uint8)")
    print("img = Image.fromarray(img_array)")
    print("img.save('gradient.png')")
    print("")
    print("# 读取图像到NumPy数组")
    print("loaded_img = Image.open('gradient.png')")
    print("loaded_array = np.array(loaded_img)")
    print("print(loaded_array.shape, loaded_array.dtype)")
    print("```")

def custom_data_types():
    """
    演示NumPy的自定义数据类型和结构化数组
    """
    print("\n"+"="*50)
    print("NumPy自定义数据类型和结构化数组")
    print("="*50)

    # 创建临时目录用于保存文件
    temp_dir = tempfile.mkdtemp()

    # 1. 创建结构化数组
    print("\n1. 创建结构化数组:")
    # 定义结构化数据类型
    dt = np.dtype([
        ('name', 'U20'),
        ('age', 'i4'),
        ('weight', 'f4'),
        ('is_student', 'b1')
    ])

    # 创建结构化数组
    people = np.array([
        ('Alice', 25, 55.0, True),
        ('Bob', 32, 75.5, False),
        ('Charlie', 18, 68.2, True)
    ], dtype=dt)

    print(f"结构化数组:\n{people}")
    print(f"数据类型: {people.dtype}")

def main():
    """
    主函数，按顺序运行所有示例
    """
    print("="*50)
    print("NumPy数据输入输出教程")
    print("="*50)
    print("\n本教程将演示NumPy的各种数据输入输出功能，包括：")
    print("1. 数组的保存和加载")
    print("2. 文本文件输入输出")
    print("3. 二进制文件输入输出")
    print("4. 与其他格式的互操作性")
    print("5. 自定义数据类型")
    
    input("\n按Enter键开始演示...")
    
    # 按顺序运行所有示例
    save_load_arrays()
    text_file_io()
    binary_file_io()
    interoperability()
    custom_data_types()
    
    print("\n"+"="*50)
    print("NumPy数据输入输出教程完成！")
    print("="*50)

if __name__ == "__main__":
    main()