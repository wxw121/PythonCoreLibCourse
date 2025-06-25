"""
Pandas数据操作和转换教程
本模块介绍Pandas库的数据操作和转换功能，包括合并、连接、分组、透视表等内容。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def merging_joining():
    """
    演示Pandas的合并和连接操作
    包括concat、merge、join等方法
    """
    print("="*50)
    print("Pandas合并和连接操作")
    print("="*50)

    # 创建示例DataFrame
    print("\n创建示例DataFrame:")

    # 创建第一个DataFrame
    df1 = pd.DataFrame({
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    }, index=[0, 1, 2, 3])
    print("DataFrame 1:")
    print(df1)

    # 创建第二个DataFrame
    df2 = pd.DataFrame({
        'A': ['A4', 'A5', 'A6', 'A7'],
        'B': ['B4', 'B5', 'B6', 'B7'],
        'C': ['C4', 'C5', 'C6', 'C7'],
        'D': ['D4', 'D5', 'D6', 'D7']
    }, index=[4, 5, 6, 7])
    print("\nDataFrame 2:")
    print(df2)

    # 创建第三个DataFrame
    df3 = pd.DataFrame({
        'A': ['A8', 'A9', 'A10', 'A11'],
        'B': ['B8', 'B9', 'B10', 'B11'],
        'C': ['C8', 'C9', 'C10', 'C11'],
        'D': ['D8', 'D9', 'D10', 'D11']
    }, index=[8, 9, 10, 11])
    print("\nDataFrame 3:")
    print(df3)

    # 1. 使用concat连接DataFrame
    print("\n1. 使用concat连接DataFrame:")

    # 按行连接（默认）
    result1 = pd.concat([df1, df2, df3])
    print("按行连接（默认）:")
    print(result1)

    # 按列连接
    result2 = pd.concat([df1, df2, df3], axis=1)
    print("\n按列连接:")
    print(result2)

    # 使用keys参数创建层次化索引
    result3 = pd.concat([df1, df2, df3], keys=['x', 'y', 'z'])
    print("\n使用keys参数创建层次化索引:")
    print(result3)

    # 2. 使用merge合并DataFrame
    print("\n2. 使用merge合并DataFrame:")

    # 创建用于合并的DataFrame
    left = pd.DataFrame({
        'key': ['K0', 'K1', 'K2', 'K3'],
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3']
    })
    right = pd.DataFrame({
        'key': ['K0', 'K1', 'K2', 'K3'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    })
    print("Left DataFrame:")
    print(left)
    print("\nRight DataFrame:")
    print(right)

    # 内连接（默认）
    result4 = pd.merge(left, right, on='key')
    print("\n内连接（默认）:")
    print(result4)

    # 创建具有重复键的DataFrame
    left2 = pd.DataFrame({
        'key1': ['K0', 'K0', 'K1', 'K2'],
        'key2': ['K0', 'K1', 'K0', 'K1'],
        'A': ['A0', 'A1', 'A2', 'A3'],
        'B': ['B0', 'B1', 'B2', 'B3']
    })
    right2 = pd.DataFrame({
        'key1': ['K0', 'K1', 'K1', 'K2'],
        'key2': ['K0', 'K0', 'K0', 'K0'],
        'C': ['C0', 'C1', 'C2', 'C3'],
        'D': ['D0', 'D1', 'D2', 'D3']
    })
    print("\nLeft DataFrame (多键):")
    print(left2)
    print("\nRight DataFrame (多键):")
    print(right2)

    # 多键合并
    result5 = pd.merge(left2, right2, on=['key1', 'key2'])
    print("\n多键合并:")
    print(result5)

    # 不同类型的连接
    print("\n不同类型的连接:")

    # 左连接
    result6 = pd.merge(left, right, on='key', how='left')
    print("左连接:")
    print(result6)

    # 右连接
    result7 = pd.merge(left, right, on='key', how='right')
    print("\n右连接:")
    print(result7)

    # 外连接
    result8 = pd.merge(left, right, on='key', how='outer')
    print("\n外连接:")
    print(result8)

    # 3. 使用join方法
    print("\n3. 使用join方法:")

    # 创建用于join的DataFrame
    left3 = pd.DataFrame({
        'A': ['A0', 'A1', 'A2'],
        'B': ['B0', 'B1', 'B2']
    }, index=['K0', 'K1', 'K2'])
    right3 = pd.DataFrame({
        'C': ['C0', 'C1', 'C2'],
        'D': ['D0', 'D1', 'D2']
    }, index=['K0', 'K2', 'K3'])
    print("Left DataFrame:")
    print(left3)
    print("\nRight DataFrame:")
    print(right3)

    # 使用join方法（默认左连接）
    result9 = left3.join(right3)
    print("\n使用join方法（默认左连接）:")
    print(result9)

    # 使用join方法（内连接）
    result10 = left3.join(right3, how='inner')
    print("\n使用join方法（内连接）:")
    print(result10)

def reshaping_data():
    """
    演示Pandas的数据重塑操作
    包括stack、unstack、pivot、melt等方法
    """
    print("\n"+"="*50)
    print("Pandas数据重塑操作")
    print("="*50)

    # 创建示例DataFrame
    print("\n创建示例DataFrame:")
    df = pd.DataFrame({
        'A': ['a', 'b', 'c', 'd'] * 3,
        'B': ['A', 'B', 'C'] * 4,
        'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
        'D': np.random.randn(12),
        'E': np.random.randn(12)
    })
    print(df)

    # 1. 使用pivot方法
    print("\n1. 使用pivot方法:")

    # 创建透视表
    pivot_table = df.pivot(index='A', columns='B', values='D')
    print("透视表 (A为索引，B为列，D为值):")
    print(pivot_table)

    # 2. 使用pivot_table方法
    print("\n2. 使用pivot_table方法:")

    # 创建透视表，可以处理重复值
    pivot_table2 = pd.pivot_table(df, values='D', index=['A', 'C'], columns=['B'])
    print("透视表 (A和C为索引，B为列，D为值):")
    print(pivot_table2)

    # 使用聚合函数
    pivot_table3 = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
                                 columns=['B'], aggfunc={'D': np.mean, 'E': np.sum})
    print("\n透视表 (使用不同的聚合函数):")
    print(pivot_table3)

    # 3. 使用stack和unstack方法
    print("\n3. 使用stack和unstack方法:")

    # 创建多级索引的DataFrame
    multi_index_df = pd.DataFrame({
        'first': ['A', 'A', 'B', 'B'],
        'second': ['one', 'two', 'one', 'two'],
        'value1': [1, 2, 3, 4],
        'value2': [10, 20, 30, 40]
    })
    multi_index_df = multi_index_df.set_index(['first', 'second'])
    print("多级索引的DataFrame:")
    print(multi_index_df)

    # 使用stack方法：将 multi_index_df（多级索引 DataFrame）从“宽格式”转换为“长格式”（堆叠）
    # 原 DataFrame 的列名会变为索引的一部分，数据变为一列
    stacked = multi_index_df.stack()
    print("\n使用stack方法:")
    print(stacked)

    # 使用unstack方法：对 stacked（已堆叠的 Series/DataFrame）执行反向操作，将其从“长格式”转换回“宽格式
    # 索引层级会重新变为列名。
    unstacked = stacked.unstack()
    print("\n使用unstack方法:")
    print(unstacked)

    # 指定级别unstack，参数1表示要取消堆叠的轴级别（level=1）
    unstacked_level1 = stacked.unstack(1)
    print("\n指定级别unstack (level=1):")
    print(unstacked_level1)

    # 4. 使用melt方法
    print("\n4. 使用melt方法:")

    # 创建用于melt的DataFrame
    df_melt = pd.DataFrame({
        'A': {0: 'a', 1: 'b', 2: 'c'},
        'B': {0: 1, 1: 3, 2: 5},
        'C': {0: 2, 1: 4, 2: 6}
    })
    print("原始DataFrame:")
    print(df_melt)

    # 使用melt方法：对数据框 df_melt 进行数据重塑，将指定的列从宽格式转为长格式。
    # 参数说明：
    #   df_melt: 要进行重塑的数据框。
    #   id_vars=['A']: 指定在转换过程中保持不变的列，这里为列 'A'。
    #   value_vars=['B', 'C']: 指定需要转换的列，这里为列 'B' 和 'C'。
    melted = pd.melt(df_melt, id_vars=['A'], value_vars=['B', 'C'])
    print("\n使用melt方法:")
    print(melted)

    # 添加自定义名称
    melted2 = pd.melt(df_melt, id_vars=['A'], value_vars=['B', 'C'],
                     var_name='变量', value_name='值')
    print("\n使用melt方法 (自定义名称):")
    print(melted2)

def groupby_operations():
    """
    演示Pandas的分组操作
    包括groupby方法及其聚合、转换、过滤等操作
    """
    print("\n"+"="*50)
    print("Pandas分组操作")
    print("="*50)

    # 创建示例DataFrame
    print("\n创建示例DataFrame:")
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
        'C': np.random.randn(8),
        'D': np.random.randn(8)
    })
    print(df)

    # 1. 基本分组操作
    print("\n1. 基本分组操作:")

    # 按单列分组
    grouped = df.groupby('A')
    print("按A列分组并计算C和D列的平均值:")
    print(grouped[['C', 'D']].mean())

    # 按多列分组
    grouped2 = df.groupby(['A', 'B'])
    print("\n按A和B列分组并计算C和D列的平均值:")
    print(grouped2[['C', 'D']].mean())

    # 2. 聚合操作
    print("\n2. 聚合操作:")

    # 使用内置聚合函数
    print("使用内置聚合函数:")
    print(grouped[['C', 'D']].agg(['mean', 'std', 'count']))

    # 对不同列使用不同的聚合函数
    print("\n对不同列使用不同的聚合函数:")
    print(grouped[['C', 'D']].agg({'C': 'sum', 'D': ['mean', 'std']}))

    # 使用自定义聚合函数
    print("\n使用自定义聚合函数:")
    print(grouped[['C', 'D']].agg(lambda x: x.max() - x.min()))

    # 3. 转换操作
    print("\n3. 转换操作:")

    # 使用transform方法
    print("使用transform方法计算每组的z-score:")
    # z-score的计算公式为：(x - 均值) / 标准差
    def zscore(x):
        return (x - x.mean()) / x.std()

    print(df.groupby('A')[['C', 'D']].transform(zscore))

    # 4. 过滤操作
    print("\n4. 过滤操作:")

    # 使用filter方法
    print("使用filter方法过滤组:")
    print(df.groupby('A').filter(lambda x: len(x) >= 3))

    # 5. 应用操作
    print("\n5. 应用操作:")

    # 使用apply方法
    print("使用apply方法应用自定义函数:")
    def top(df, n=2, column='C'):
        return df.sort_values(by=column, ascending=False)[:n]

    print(df.groupby('A').apply(top))

    # 6. 分组迭代
    print("\n6. 分组迭代:")

    # 遍历分组
    print("遍历分组:")
    for name, group in df.groupby('A'):
        print(f"\n组名: {name}")
        print(group)

def time_series_operations():
    """
    演示Pandas的时间序列操作
    包括日期范围生成、重采样、移动窗口等操作
    """
    print("\n"+"="*50)
    print("Pandas时间序列操作")
    print("="*50)
    
    # 1. 创建时间序列
    print("\n1. 创建时间序列:")
    
    # 创建日期范围
    date_range = pd.date_range(start='2023-01-01', end='2023-01-10')
    print("日期范围:")
    print(date_range)
    
    # 创建指定周期的日期范围
    date_range2 = pd.date_range(start='2023-01-01', periods=10, freq='D')
    print("\n指定周期的日期范围:")
    print(date_range2)
    
    # 创建不同频率的日期范围
    print("\n不同频率的日期范围:")
    print("小时频率:")
    print(pd.date_range(start='2023-01-01', periods=5, freq='H'))
    print("\n工作日频率:")
    print(pd.date_range(start='2023-01-01', periods=5, freq='B'))
    print("\n月频率:")
    print(pd.date_range(start='2023-01-01', periods=5, freq='M'))
    print("\n季度频率:")
    print(pd.date_range(start='2023-01-01', periods=5, freq='Q'))
    
    # 2. 时间序列索引
    print("\n2. 时间序列索引:")
    
    # 创建带有时间索引的Series
    ts = pd.Series(np.random.randn(10), index=date_range)
    print("带有时间索引的Series:")
    print(ts)
    
    # 时间索引的切片操作
    print("\n时间索引的切片操作:")
    print(ts['2023-01-03':'2023-01-07'])
    
    # 3. 重采样
    print("\n3. 重采样:")
    
    # 创建更多数据点
    ts_long = pd.Series(np.random.randn(100), 
                        index=pd.date_range('2023-01-01', periods=100, freq='H'))
    
    # 上采样
    print("上采样 (小时 -> 分钟):")
    print(ts_long.resample('15T').mean().head())
    
    # 下采样
    print("\n下采样 (小时 -> 天):")
    print(ts_long.resample('D').mean())
    
    # 4. 移动窗口函数
    print("\n4. 移动窗口函数:")
    
    # 创建示例Series
    s = pd.Series(np.random.randn(10), index=date_range)
    print("原始Series:")
    print(s)

    '''
    移动平均：
    1. `s.rolling(window=3)` 创建一个滚动窗口对象，窗口大小为3
    2. `.mean()` 计算每个窗口内的平均值
    3. 整体操作会生成一个新的序列，其中每个元素是原序列对应位置前3个元素(包括自己)的平均值
    4. 第一个和第二个元素由于没有足够的前驱数据，结果会是NaN(代码中未显示这部分输出)
    '''
    print("\n移动平均 (窗口大小=3):")
    print(s.rolling(window=3).mean())
    
    # 扩展窗口: 计算从序列起始点到当前位置的累积平均值
    # `expanding()`创建一个动态增长的窗口（初始包含第1个元素，逐步扩展至全部元素），`.mean()`计算每个窗口状态的算术平均值。常用于时间序列的累积统计分析。
    print("\n扩展窗口:")
    print(s.expanding().mean())
    
    # 指数加权移动平均
    print("\n指数加权移动平均:")
    print(s.ewm(span=3).mean())
    
    # 5. 时区处理
    print("\n5. 时区处理:")
    
    # 创建带有时区的时间序列
    ts_utc = pd.Series(np.random.randn(5), 
                      index=pd.date_range('2023-01-01', periods=5, tz='UTC'))
    print("UTC时区的时间序列:")
    print(ts_utc)
    
    # 转换时区
    print("\n转换为美国东部时间:")
    print(ts_utc.tz_convert('US/Eastern'))
    
    # 本地化时区
    ts_naive = pd.Series(np.random.randn(5), 
                        index=pd.date_range('2023-01-01', periods=5))
    print("\n本地化时区:")
    print(ts_naive.tz_localize('Asia/Shanghai'))

def main():
    """
    主函数，按顺序运行所有示例
    """
    print("="*50)
    print("Pandas数据操作和转换教程")
    print("="*50)
    print("\n本教程将介绍以下内容：")
    print("1. 合并和连接操作")
    print("2. 数据重塑操作")
    print("3. 分组操作")
    print("4. 时间序列操作")
    
    input("\n按Enter键开始演示...")
    
    # 运行所有示例
    merging_joining()
    reshaping_data()
    groupby_operations()
    time_series_operations()
    
    print("\n"+"="*50)
    print("Pandas数据操作和转换教程完成！")
    print("="*50)

if __name__ == "__main__":
    main()