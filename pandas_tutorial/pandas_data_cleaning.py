"""
Pandas数据清洗和预处理教程
本模块介绍Pandas库的数据清洗和预处理功能，包括处理缺失值、重复值、异常值等内容。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def handling_missing_values():
    """
    演示Pandas处理缺失值的方法
    包括检测、填充、删除缺失值等操作
    """
    print("="*50)
    print("Pandas处理缺失值")
    print("="*50)
    
    # 创建包含缺失值的DataFrame
    print("\n创建包含缺失值的DataFrame:")
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 2, 3, np.nan, np.nan],
        'D': ['a', 'b', 'c', None, 'e']
    })
    print(df)
    
    # 1. 检测缺失值
    print("\n1. 检测缺失值:")
    
    # 使用isna()检测缺失值
    print("使用isna()检测缺失值:")
    print(df.isna())
    
    # 使用notna()检测非缺失值
    print("\n使用notna()检测非缺失值:")
    print(df.notna())

    # 检查每列的缺失值数量
    print("\n检查每列的缺失值数量:")
    print(df.isna().sum())

    # 检查每行的缺失值数量
    print("\n检查每行的缺失值数量:")
    print(df.isna().sum(axis=1))

    # 检查是否存在任何缺失值
    print("\n检查是否存在任何缺失值:")
    '''
    df.isna() 检查DataFrame中每个元素是否为缺失值(NaN)，返回一个布尔型DataFrame
    .any() 第一次调用，检查每列是否有至少一个True(即是否有缺失值)，返回一个布尔型Series
    .any() 第二次调用，检查整个Series是否有至少一个True，最终返回一个布尔值(True/False)
    '''
    print(f"DataFrame中是否存在缺失值: {df.isna().any().any()}")

    # 2. 删除缺失值
    print("\n2. 删除缺失值:")

    # 删除包含任何缺失值的行
    print("删除包含任何缺失值的行:")
    print(df.dropna())

    # 删除所有值都是缺失值的行
    print("\n删除所有值都是缺失值的行:")
    print(df.dropna(how='all'))

    # 删除包含任何缺失值的列
    print("\n删除包含任何缺失值的列:")
    print(df.dropna(axis=1))

    # 删除所有值都是缺失值的列
    print("\n删除所有值都是缺失值的列:")
    print(df.dropna(axis=1, how='all'))

    # 删除缺失值超过阈值的行
    print("\n删除缺失值超过阈值的行 (至少2个非缺失值):")
    print(df.dropna(thresh=2))

    # 3. 填充缺失值
    print("\n3. 填充缺失值:")

    # 使用常数填充
    print("使用常数填充:")
    print(df.fillna(0))

    # 使用不同的值填充不同的列
    print("\n使用不同的值填充不同的列:")
    print(df.fillna({'A': 0, 'B': 1, 'C': 2, 'D': 'missing'}))

    # 使用前向填充
    print("\n使用前向填充:")
    print(df.fillna(method='ffill'))

    # 使用后向填充
    print("\n使用后向填充:")
    print(df.fillna(method='bfill'))

    # 限制填充次数
    print("\n限制填充次数 (最多填充1次):")
    df_with_more_nans = pd.DataFrame({
        'A': [1, np.nan, np.nan, np.nan, 5],
        'B': [np.nan, 2, np.nan, np.nan, 5]
    })
    print("原始DataFrame:")
    print(df_with_more_nans)
    print("\n前向填充 (limit=1):")
    print(df_with_more_nans.fillna(method='ffill', limit=1))

    # 使用统计值填充
    print("\n使用统计值填充:")
    print("使用平均值填充:")
    # 只对数值列应用平均值填充
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_filled = df.copy()
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
    print(df_filled)

    # 4. 插值
    print("\n4. 插值:")

    # 创建用于插值的DataFrame
    df_interp = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8],
        'B': [np.nan, 2, 3, 4, np.nan, 6, 7, 8]
    })
    print("原始DataFrame:")
    print(df_interp)

    # 线性插值
    print("\n线性插值:")
    print(df_interp.interpolate())

    # 不同的插值方法
    print("\n不同的插值方法:")
    print("最近邻插值:")
    print(df_interp.interpolate(method='nearest'))
    print("\n二次样条插值:")
    print(df_interp.interpolate(method='quadratic'))

    # 5. 替换特定值
    print("\n5. 替换特定值:")

    # 创建包含特定值的DataFrame
    df_replace = pd.DataFrame({
        'A': [1, 2, -999, 4, 5],
        'B': [-999, 2, 3, 4, 5],
        'C': [1, 2, 3, -999, -999]
    })
    print("原始DataFrame (其中-999表示缺失值):")
    print(df_replace)

    # 替换特定值
    print("\n替换特定值:")
    print(df_replace.replace(-999, np.nan))

    # 替换多个值
    print("\n替换多个值:")
    print(df_replace.replace([-999, 2], [np.nan, 200]))

    # 使用字典替换
    print("\n使用字典替换:")
    print(df_replace.replace({-999: np.nan, 2: 200}))

def handling_duplicates():
    """
    演示Pandas处理重复值的方法
    包括检测、删除重复值等操作
    """
    print("\n"+"="*50)
    print("Pandas处理重复值")
    print("="*50)

    # 创建包含重复值的DataFrame
    print("\n创建包含重复值的DataFrame:")
    df = pd.DataFrame({
        'A': [1, 1, 2, 3, 4, 4],
        'B': [5, 5, 6, 7, 8, 8],
        'C': ['a', 'a', 'b', 'c', 'd', 'd']
    })
    print(df)

    # 1. 检测重复值
    print("\n1. 检测重复值:")

    # 检测完全重复的行
    print("检测完全重复的行:")
    print(df.duplicated())

    # 检测特定列的重复值
    print("\n检测特定列的重复值:")
    print(df.duplicated(subset=['A', 'B']))

    # 保留最后一个重复项
    print("\n保留最后一个重复项:")
    print(df.duplicated(keep='last'))

    # 标记所有重复项
    print("\n标记所有重复项:")
    print(df.duplicated(keep=False))

    # 2. 删除重复值
    print("\n2. 删除重复值:")

    # 删除完全重复的行
    print("删除完全重复的行:")
    print(df.drop_duplicates())

    # 删除特定列的重复值
    print("\n删除特定列的重复值:")
    print(df.drop_duplicates(subset=['A', 'B']))

    # 保留最后一个重复项
    print("\n保留最后一个重复项:")
    print(df.drop_duplicates(keep='last'))

    # 删除所有重复项
    print("\n删除所有重复项 (不保留任何重复项):")
    print(df.drop_duplicates(keep=False))

    # 3. 计数和汇总重复值
    print("\n3. 计数和汇总重复值:")

    # 计算每个值的出现次数
    print("计算每个值的出现次数:")
    print(df['A'].value_counts())

    # 计算每个值的出现次数 (包括缺失值)
    print("\n计算每个值的出现次数 (包括缺失值):")
    df_with_nan = pd.DataFrame({
        'A': [1, 1, 2, np.nan, np.nan],
        'B': [5, 5, 6, 7, 7]
    })
    print(df_with_nan)
    print("\n值计数:")
    print(df_with_nan['A'].value_counts(dropna=False))

    # 计算重复值的百分比
    print("\n计算重复值的百分比:")
    print(df['A'].value_counts(normalize=True))

def handling_outliers():
    """
    演示Pandas处理异常值的方法
    包括检测、过滤、替换异常值等操作
    """
    print("\n"+"="*50)
    print("Pandas处理异常值")
    print("="*50)

    # 创建包含异常值的DataFrame
    print("\n创建包含异常值的DataFrame:")
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),  # 正态分布数据
        'B': np.random.normal(0, 1, 100),
        'C': np.random.normal(0, 1, 100)
    })

    # 添加一些异常值
    df.loc[10, 'A'] = 10  # 远离均值的异常值
    df.loc[20, 'B'] = -10
    df.loc[30, 'C'] = 15
    df.loc[40, 'A'] = -12

    print(df.describe())

    # 1. 使用统计方法检测异常值
    print("\n1. 使用统计方法检测异常值:")

    # 使用Z-score检测异常值
    print("使用Z-score检测异常值:")
    z_scores = (df - df.mean()) / df.std()
    print("Z-scores:")
    print(z_scores.head())

    # 标记Z-score绝对值大于3的值为异常值
    print("\n标记Z-score绝对值大于3的值为异常值:")
    outliers_z = (abs(z_scores) > 3).any(axis=1)
    print("异常值的行索引:")
    print(df[outliers_z].index.tolist())

    # 使用IQR (四分位距) 检测异常值
    print("\n使用IQR (四分位距) 检测异常值:")
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # 定义异常值边界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("下界:")
    print(lower_bound)
    print("\n上界:")
    print(upper_bound)

    # 标记超出边界的值为异常值
    outliers_iqr = ((df < lower_bound) | (df > upper_bound)).any(axis=1)
    print("\n异常值的行索引:")
    print(df[outliers_iqr].index.tolist())

    # 2. 过滤异常值
    print("\n2. 过滤异常值:")

    # 移除Z-score异常值
    print("移除Z-score异常值:")
    df_no_outliers_z = df[~outliers_z]
    print(f"原始数据形状: {df.shape}, 移除异常值后: {df_no_outliers_z.shape}")

    # 移除IQR异常值
    print("\n移除IQR异常值:")
    df_no_outliers_iqr = df[~outliers_iqr]
    print(f"原始数据形状: {df.shape}, 移除异常值后: {df_no_outliers_iqr.shape}")

    # 3. 替换异常值
    print("\n3. 替换异常值:")

    # 将异常值替换为边界值
    print("将异常值替换为边界值:")
    df_capped = df.copy()

    for column in df_capped.columns:
        lower = lower_bound[column]
        upper = upper_bound[column]
        df_capped[column] = df_capped[column].clip(lower=lower, upper=upper)

    print("原始数据:")
    print(df.describe())
    print("\n替换异常值后:")
    print(df_capped.describe())

    # 将异常值替换为NaN
    print("\n将异常值替换为NaN:")
    df_nan = df.copy()
    df_nan[((df < lower_bound) | (df > upper_bound))] = np.nan
    print(df_nan.head(50))

    # 4. 可视化异常值
    print("\n4. 可视化异常值 (使用箱线图):")
    print("请运行以下代码查看箱线图:")
    print("plt.figure(figsize=(10, 6))")
    print("df.boxplot()")
    print("plt.title('箱线图展示异常值')")
    print("plt.show()")

def data_transformation():
    """
    演示Pandas的数据转换方法
    包括类型转换、标准化、编码等操作
    """
    print("\n"+"="*50)
    print("Pandas数据转换")
    print("="*50)
    
    # 创建示例DataFrame
    print("\n创建示例DataFrame:")
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10.1, 20.2, 30.3, 40.4, 50.5],
        'C': ['a', 'b', 'c', 'd', 'e'],
        'D': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05'],
        'E': ['True', 'False', 'True', 'False', 'True']
    })
    print(df)
    print("\n数据类型:")
    print(df.dtypes)
    
    # 1. 类型转换
    print("\n1. 类型转换:")
    
    # 转换单列
    print("转换单列:")
    df['A'] = df['A'].astype(float)
    print(f"A列转换为float: {df['A'].dtype}")
    
    # 转换多列
    print("\n转换多列:")
    # 先将字符串'True'/'False'转换为实际的布尔值
    df['E'] = df['E'].map({'True': True, 'False': False})
    df = df.astype({'B': int, 'E': bool})
    print("转换后的数据类型:")
    print(df.dtypes)
    
    # 转换日期列
    print("\n转换日期列:")
    df['D'] = pd.to_datetime(df['D'])
    print(f"D列转换为datetime: {df['D'].dtype}")
    print(df['D'])
    
    # 2. 标准化和归一化
    print("\n2. 标准化和归一化:")
    
    # 创建用于标准化的DataFrame
    df_norm = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })
    print("原始数据:")
    print(df_norm)
    
    # 最小-最大归一化
    print("\n最小-最大归一化:")
    min_max_scaled = (df_norm - df_norm.min()) / (df_norm.max() - df_norm.min())
    print(min_max_scaled)
    
    # Z-score标准化
    print("\nZ-score标准化:")
    z_scaled = (df_norm - df_norm.mean()) / df_norm.std()
    print(z_scaled)
    
    # 使用sklearn进行标准化
    print("\n使用sklearn进行标准化 (示例代码):")
    print("from sklearn.preprocessing import StandardScaler")
    print("scaler = StandardScaler()")
    print("scaled_data = scaler.fit_transform(df_norm)")
    print("scaled_df = pd.DataFrame(scaled_data, columns=df_norm.columns)")
    
    # 3. 编码分类变量
    print("\n3. 编码分类变量:")
    
    # 创建包含分类变量的DataFrame
    df_cat = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red', 'blue'],
        'size': ['small', 'medium', 'large', 'medium', 'small'],
        'price': [10, 20, 30, 15, 25]
    })
    print("原始数据:")
    print(df_cat)
    
    # 独热编码
    print("\n独热编码:")
    dummies = pd.get_dummies(df_cat[['color', 'size']])
    print(dummies)
    
    # 合并独热编码结果
    print("\n合并独热编码结果:")
    df_encoded = pd.concat([df_cat['price'], dummies], axis=1)
    print(df_encoded)
    
    # 标签编码
    print("\n标签编码:")
    df_cat['color_code'] = pd.factorize(df_cat['color'])[0]
    df_cat['size_code'] = pd.factorize(df_cat['size'])[0]
    print(df_cat)
    
    # 4. 字符串操作
    print("\n4. 字符串操作:")
    
    # 创建包含字符串的DataFrame
    df_str = pd.DataFrame({
        'text': ['Python 3.9', 'pandas 1.3.0', 'numpy 1.20.0', 'matplotlib 3.4.2', 'scikit-learn 0.24.2']
    })
    print("原始数据:")
    print(df_str)
    
    # 提取库名称
    print("\n提取库名称:")
    df_str['library'] = df_str['text'].str.split(' ').str[0]
    print(df_str)
    
    # 提取版本号
    print("\n提取版本号:")
    df_str['version'] = df_str['text'].str.split(' ').str[1]
    print(df_str)
    
    # 字符串替换
    print("\n字符串替换:")
    df_str['text_replaced'] = df_str['text'].str.replace(r'\d+\.\d+\.\d+|\d+\.\d+', 'X.Y.Z', regex=True)
    print(df_str)
    
    # 字符串包含
    print("\n字符串包含:")
    df_str['contains_py'] = df_str['text'].str.contains('py', case=False)
    print(df_str)

def main():
    """
    主函数，按顺序运行所有示例
    """
    print("="*50)
    print("Pandas数据清洗和预处理教程")
    print("="*50)
    print("\n本教程将介绍以下内容：")
    print("1. 处理缺失值")
    print("2. 处理重复值")
    print("3. 处理异常值")
    print("4. 数据转换")
    
    input("\n按Enter键开始演示...")
    
    # 运行所有示例
    handling_missing_values()
    handling_duplicates()
    handling_outliers()
    data_transformation()
    
    print("\n"+"="*50)
    print("Pandas数据清洗和预处理教程完成！")
    print("="*50)

if __name__ == "__main__":
    main()