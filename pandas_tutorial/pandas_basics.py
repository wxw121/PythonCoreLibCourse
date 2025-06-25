"""
Pandas基础知识教程
本模块介绍Pandas库的基础知识，包括Series和DataFrame的创建、基本操作和属性等内容。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def series_basics():
    """
    演示Pandas Series的基础知识
    Series是带有标签的一维数组，可以保存任何数据类型
    """
    print("="*50)
    print("Pandas Series基础知识")
    print("="*50)

    # 1. 创建Series
    print("\n1. 创建Series:")

    # 从列表创建Series
    s1 = pd.Series([1, 3, 5, 7, 9])
    print("从列表创建Series:")
    print(s1)

    # 从NumPy数组创建Series
    s2 = pd.Series(np.array([2, 4, 6, 8, 10]))
    print("\n从NumPy数组创建Series:")
    print(s2)

    # 创建带有自定义索引的Series
    s3 = pd.Series([100, 200, 300, 400], index=['a', 'b', 'c', 'd'])
    print("\n创建带有自定义索引的Series:")
    print(s3)

    # 从字典创建Series
    s4 = pd.Series({'a': 100, 'b': 200, 'c': 300})
    print("\n从字典创建Series:")
    print(s4)

    # 创建带有标量值的Series
    s5 = pd.Series(5, index=['a', 'b', 'c'])
    print("\n创建带有标量值的Series:")
    print(s5)

    # 2. Series的属性
    print("\n2. Series的属性:")
    print(f"值: {s3.values}")  # 获取Series的值
    print(f"索引: {s3.index}")  # 获取Series的索引
    print(f"数据类型: {s3.dtype}")  # 获取Series的数据类型
    print(f"形状: {s3.shape}")  # 获取Series的形状
    print(f"大小: {s3.size}")  # 获取Series的大小
    print(f"维度: {s3.ndim}")  # 获取Series的维度

    # 3. 访问Series元素
    print("\n3. 访问Series元素:")
    print(f"通过位置访问 (s3[2]): {s3[2]}")  # 通过位置访问
    print(f"通过标签访问 (s3['c']): {s3['c']}")  # 通过标签访问
    print(f"通过位置切片 (s3[1:3]):\n{s3[1:3]}")  # 通过位置切片
    print(f"通过标签切片 (s3['b':'c']):\n{s3['b':'c']}")  # 通过标签切片

    # 4. Series的基本操作
    print("\n4. Series的基本操作:")

    # 算术运算
    print("算术运算:")
    print(f"s3 + 5:\n{s3 + 5}")  # 加法
    print(f"s3 * 2:\n{s3 * 2}")  # 乘法

    # Series之间的运算
    print("\nSeries之间的运算:")
    s6 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
    print(f"s3:\n{s3}")
    print(f"s6:\n{s6}")
    print(f"s3 + s6:\n{s3 + s6}")  # Series相加

    # 不同索引的Series运算
    s7 = pd.Series([10, 20, 30], index=['a', 'b', 'e'])
    print("\n不同索引的Series运算:")
    print(f"s3:\n{s3}")
    print(f"s7:\n{s7}")
    print(f"s3 + s7:\n{s3 + s7}")  # 不同索引的Series相加，不匹配的索引结果为NaN

    # 5. Series的方法
    print("\n5. Series的方法:")

    # 统计方法
    print("统计方法:")
    s8 = pd.Series([10, 20, 30, 40, 50])
    print(f"s8:\n{s8}")
    print(f"求和 (sum): {s8.sum()}")
    print(f"平均值 (mean): {s8.mean()}")
    print(f"最大值 (max): {s8.max()}")
    print(f"最小值 (min): {s8.min()}")
    print(f"标准差 (std): {s8.std()}")
    print(f"描述性统计 (describe):\n{s8.describe()}")

    # 应用函数
    print("\n应用函数:")
    print(f"应用平方函数 (apply):\n{s8.apply(lambda x: x**2)}")

    # 排序
    print("\n排序:")
    s9 = pd.Series([3, 1, 4, 2], index=['d', 'b', 'a', 'c'])
    print(f"原始Series:\n{s9}")
    print(f"按值排序 (sort_values):\n{s9.sort_values()}")
    print(f"按索引排序 (sort_index):\n{s9.sort_index()}")

def dataframe_basics():
    """
    演示Pandas DataFrame的基础知识
    DataFrame是带有标签的二维表格数据结构，类似于电子表格或SQL表
    """
    print("\n"+"="*50)
    print("Pandas DataFrame基础知识")
    print("="*50)

    # 1. 创建DataFrame
    print("\n1. 创建DataFrame:")

    # 从字典创建DataFrame
    df1 = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40],
        'C': ['a', 'b', 'c', 'd']
    })
    print("从字典创建DataFrame:")
    print(df1)

    # 从嵌套列表创建DataFrame
    df2 = pd.DataFrame(
        [[1, 10, 'a'],
         [2, 20, 'b'],
         [3, 30, 'c'],
         [4, 40, 'd']],
        columns=['A', 'B', 'C']
    )
    print("\n从嵌套列表创建DataFrame:")
    print(df2)

    # 从Series字典创建DataFrame
    s1 = pd.Series([1, 2, 3, 4])
    s2 = pd.Series([10, 20, 30, 40])
    s3 = pd.Series(['a', 'b', 'c', 'd'])
    df3 = pd.DataFrame({'A': s1, 'B': s2, 'C': s3})
    print("\n从Series字典创建DataFrame:")
    print(df3)

    # 从NumPy数组创建DataFrame
    df4 = pd.DataFrame(
        np.random.randn(4, 3),  # 创建4x3的随机数组
        columns=['A', 'B', 'C'],
        index=['row1', 'row2', 'row3', 'row4']
    )
    print("\n从NumPy数组创建DataFrame:")
    print(df4)

    # 2. DataFrame的属性
    print("\n2. DataFrame的属性:")
    print(f"列: {df1.columns}")  # 获取列名
    print(f"索引: {df1.index}")  # 获取行索引
    print(f"值: \n{df1.values}")  # 获取数据值
    print(f"形状: {df1.shape}")  # 获取形状
    print(f"大小: {df1.size}")  # 获取元素总数
    print(f"维度: {df1.ndim}")  # 获取维度
    print(f"数据类型: \n{df1.dtypes}")  # 获取每列的数据类型

    # 3. 访问DataFrame元素
    print("\n3. 访问DataFrame元素:")

    # 访问列
    print("访问列:")
    print(f"通过列名访问 (df1['A']):\n{df1['A']}")  # 通过列名访问
    print(f"通过属性访问 (df1.A):\n{df1.A}")  # 通过属性访问
    print(f"访问多列 (df1[['A', 'C']]):\n{df1[['A', 'C']]}")  # 访问多列

    # 访问行
    print("\n访问行:")
    print(f"通过位置访问 (df1.iloc[1]):\n{df1.iloc[1]}")  # 通过位置访问
    print(f"通过标签访问 (df4.loc['row2']):\n{df4.loc['row2']}")  # 通过标签访问

    # 访问特定元素
    print("\n访问特定元素:")
    print(f"通过位置访问元素 (df1.iloc[0, 1]): {df1.iloc[0, 1]}")  # 通过位置访问元素
    print(f"通过标签访问元素 (df4.loc['row1', 'B']): {df4.loc['row1', 'B']}")  # 通过标签访问元素

    # 切片
    print("\n切片:")
    print(f"行切片 (df1.iloc[1:3]):\n{df1.iloc[1:3]}")  # 行切片
    print(f"行列切片 (df1.iloc[1:3, 0:2]):\n{df1.iloc[1:3, 0:2]}")  # 行列切片

    # 条件选择
    print("\n条件选择:")
    print(f"选择A列大于2的行:\n{df1[df1['A'] > 2]}")  # 条件选择
    print(f"选择A列大于2且B列小于40的行:\n{df1[(df1['A'] > 2) & (df1['B'] < 40)]}")  # 复合条件选择

    # 4. DataFrame的基本操作
    print("\n4. DataFrame的基本操作:")

    # 添加列
    df5 = df1.copy()
    df5['D'] = [100, 200, 300, 400]
    print("添加列:")
    print(df5)

    # 删除列
    df6 = df5.copy()
    df6 = df6.drop('D', axis=1)  # axis=1表示列
    print("\n删除列:")
    print(df6)

    # 添加行
    df7 = df1.copy()
    df7.loc[4] = [5, 50, 'e']
    print("\n添加行:")
    print(df7)

    # 删除行
    df8 = df7.copy()
    df8 = df8.drop(4, axis=0)  # axis=0表示行
    print("\n删除行:")
    print(df8)

    # 5. DataFrame的方法
    print("\n5. DataFrame的方法:")

    # 统计方法
    print("统计方法:")
    df9 = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    print(f"DataFrame:\n{df9}")
    print(f"求和 (sum):\n{df9.sum()}")  # 默认按列求和
    print(f"按行求和 (sum(axis=1)):\n{df9.sum(axis=1)}")  # 按行求和
    print(f"平均值 (mean):\n{df9.mean()}")
    print(f"描述性统计 (describe):\n{df9.describe()}")

    # 应用函数
    print("\n应用函数:")
    print(f"对每个元素应用函数 (applymap):\n{df9.applymap(lambda x: x*2)}")  # 对每个元素应用函数
    print(f"对每列应用函数 (apply):\n{df9.apply(lambda x: x.max() - x.min())}")  # 对每列应用函数

    # 排序
    print("\n排序:")
    df10 = pd.DataFrame({
        'A': [3, 1, 4, 2],
        'B': [10, 40, 20, 30],
        'C': ['a', 'd', 'b', 'c']
    })
    print(f"原始DataFrame:\n{df10}")
    print(f"按A列排序 (sort_values('A')):\n{df10.sort_values('A')}")  # 按A列排序
    print(f"按多列排序 (sort_values(['A', 'B'])):\n{df10.sort_values(['A', 'B'])}")  # 按多列排序
    print(f"按索引排序 (sort_index()):\n{df10.sort_index()}")  # 按索引排序

def indexing_and_selection():
    """
    演示Pandas的索引和选择操作
    包括标签索引、位置索引、布尔索引等
    """
    print("\n"+"="*50)
    print("Pandas索引和选择操作")
    print("="*50)

    # 创建示例DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500],
        'D': ['a', 'b', 'c', 'd', 'e']
    }, index=['row1', 'row2', 'row3', 'row4', 'row5'])

    print("示例DataFrame:")
    print(df)

    # 1. 标签索引 (.loc)
    print("\n1. 标签索引 (.loc):")
    print("单个标签:")
    print(f"选择'row2'行:\n{df.loc['row2']}")
    
    print("\n多个标签:")
    print(f"选择'row1'和'row3'行:\n{df.loc[['row1', 'row3']]}")
    
    print("\n标签切片:")
    print(f"选择'row2'到'row4'的行:\n{df.loc['row2':'row4']}")
    
    print("\n选择特定行和列:")
    print(f"选择'row1'行的'A'和'C'列:\n{df.loc['row1', ['A', 'C']]}")
    
    # 2. 位置索引 (.iloc)
    print("\n2. 位置索引 (.iloc):")
    print("单个位置:")
    print(f"选择第2行:\n{df.iloc[1]}")
    
    print("\n多个位置:")
    print(f"选择第1和第3行:\n{df.iloc[[0, 2]]}")
    
    print("\n位置切片:")
    print(f"选择前3行:\n{df.iloc[0:3]}")
    
    print("\n选择特定行和列的位置:")
    print(f"选择第1行的第0和第2列:\n{df.iloc[0, [0, 2]]}")
    
    # 3. 布尔索引
    print("\n3. 布尔索引:")
    print("使用单个条件:")
    print(f"A列大于2的行:\n{df[df['A'] > 2]}")
    
    print("\n使用多个条件:")
    print(f"A列大于2且B列小于40的行:\n{df[(df['A'] > 2) & (df['B'] < 40)]}")
    
    print("\n使用isin方法:")
    print(f"D列值在['a', 'c', 'e']中的行:\n{df[df['D'].isin(['a', 'c', 'e'])]}")
    
    # 4. 混合索引
    print("\n4. 混合索引:")
    print("使用query方法:")
    print(f"使用query选择A大于2的行:\n{df.query('A > 2')}")
    
    print("\n使用at访问单个值:")
    print(f"使用at访问'row2'行'B'列的值: {df.at['row2', 'B']}")
    
    print("\n使用iat访问单个值:")
    print(f"使用iat访问第1行第1列的值: {df.iat[1, 1]}")

def main():
    """
    主函数，按顺序运行所有示例
    """
    print("="*50)
    print("Pandas基础知识教程")
    print("="*50)
    print("\n本教程将介绍以下内容：")
    print("1. Series基础知识")
    print("2. DataFrame基础知识")
    print("3. 索引和选择操作")
    
    input("\n按Enter键开始演示...")
    
    # 运行所有示例
    series_basics()
    dataframe_basics()
    indexing_and_selection()
    
    print("\n"+"="*50)
    print("Pandas基础知识教程完成！")
    print("="*50)

if __name__ == "__main__":
    main()