"""
Scikit-learn 数据预处理

本模块介绍scikit-learn中的数据预处理技术，包括特征缩放、特征编码、缺失值处理和特征选择等。
数据预处理是机器学习流程中的关键步骤，可以显著提高模型的性能和稳定性。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder,
    PolynomialFeatures
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from scikit_learn_tutorial.config import set_matplotlib_chinese


def feature_scaling():
    """
    演示不同的特征缩放方法
    """
    print("=" * 50)
    print("特征缩放".center(50))
    print("=" * 50)

    # 创建示例数据
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=10, size=(100, 3))
    data[:, 1] = data[:, 1] * 100  # 放大第二列
    data[:, 2] = data[:, 2] / 100  # 缩小第三列

    # 添加一些异常值
    data[0, 0] = 100
    data[1, 1] = 1000
    data[2, 2] = -10

    # 转换为DataFrame以便于可视化
    df = pd.DataFrame(data, columns=['特征1', '特征2', '特征3'])

    print("\n原始数据统计:")
    print(df.describe().round(2))

    # 1. 标准化 (StandardScaler)
    print("\n1. 标准化 (StandardScaler)")
    print("将特征转换为均值为0，标准差为1的分布")
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    df_std = pd.DataFrame(data_std, columns=['特征1', '特征2', '特征3'])

    print("\n标准化后的数据统计:")
    print(df_std.describe().round(2))
    print(f"均值接近0: {np.mean(data_std, axis=0).round(2)}")
    print(f"标准差接近1: {np.std(data_std, axis=0).round(2)}")

    # 2. 最小-最大缩放 (MinMaxScaler)
    print("\n2. 最小-最大缩放 (MinMaxScaler)")
    print("将特征缩放到指定范围，默认为[0, 1]")
    min_max_scaler = MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(data)
    df_minmax = pd.DataFrame(data_minmax, columns=['特征1', '特征2', '特征3'])

    print("\n最小-最大缩放后的数据统计:")
    print(df_minmax.describe().round(2))
    print(f"最小值接近0: {np.min(data_minmax, axis=0).round(2)}")
    print(f"最大值接近1: {np.max(data_minmax, axis=0).round(2)}")

    # 3. 稳健缩放 (RobustScaler)
    print("\n3. 稳健缩放 (RobustScaler)")
    print("使用中位数和四分位距进行缩放，对异常值不敏感")
    robust_scaler = RobustScaler()
    data_robust = robust_scaler.fit_transform(data)
    df_robust = pd.DataFrame(data_robust, columns=['特征1', '特征2', '特征3'])

    print("\n稳健缩放后的数据统计:")
    print(df_robust.describe().round(2))

    # 可视化不同缩放方法的效果
    plt.figure(figsize=(15, 10))

    # 原始数据
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df)
    plt.title("原始数据")

    # 标准化
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df_std)
    plt.title("标准化 (StandardScaler)")

    # 最小-最大缩放
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df_minmax)
    plt.title("最小-最大缩放 (MinMaxScaler)")

    # 稳健缩放
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df_robust)
    plt.title("稳健缩放 (RobustScaler)")

    plt.tight_layout()

    # 确保目录存在
    os.makedirs("images", exist_ok=True)

    plt.savefig("images/scaling_comparison.png")
    plt.close()

    print("\n不同缩放方法的比较图已保存为 'scaling_comparison.png'")

    # 何时使用不同的缩放方法
    print("\n何时使用不同的缩放方法:")
    print("- StandardScaler: 当数据近似正态分布，且没有明显异常值时")
    print("- MinMaxScaler: 当需要特定范围的特征值，如图像处理中的像素值")
    print("- RobustScaler: 当数据包含异常值时")


def categorical_encoding():
    """
    演示不同的分类特征编码方法
    """
    print("\n" + "=" * 50)
    print("分类特征编码".center(50))
    print("=" * 50)

    # 创建示例数据
    data = {
        '颜色': ['红', '绿', '蓝', '绿', '红', '蓝', '红'],
        '尺寸': ['小', '中', '大', '小', '大', '中', '中'],
        '等级': ['低', '中', '高', '中', '高', '低', '中']
    }
    df = pd.DataFrame(data)

    print("\n原始分类数据:")
    print(df)

    # 1. 标签编码 (LabelEncoder)
    print("\n1. 标签编码 (LabelEncoder)")
    print("将每个类别映射为一个整数")

    label_encoder = LabelEncoder()
    df_label = df.copy()

    for column in df.columns:
        df_label[column] = label_encoder.fit_transform(df[column])

    print("\n标签编码后的数据:")
    print(df_label)
    print("\n注意: 标签编码可能引入序数关系，不适合没有顺序关系的类别")

    # 2. 独热编码 (OneHotEncoder)
    print("\n2. 独热编码 (OneHotEncoder)")
    print("为每个类别创建一个二进制特征")

    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(df)

    # 获取所有特征名称
    feature_names = []
    for i, column in enumerate(df.columns):
        feature_names.extend([f"{column}_{cat}" for cat in
                             onehot_encoder.categories_[i]])

    df_onehot = pd.DataFrame(onehot_encoded, columns=feature_names)

    print("\n独热编码后的数据:")
    print(df_onehot)
    print("\n注意: 独热编码会增加特征数量，可能导致维度灾难")

    # 3. 序数编码 (OrdinalEncoder)
    print("\n3. 序数编码 (OrdinalEncoder)")
    print("将类别映射为有序整数，适用于有序类别")

    # 定义类别顺序
    categories = [
        ['小', '中', '大'],  # 尺寸顺序
        ['低', '中', '高'],  # 等级顺序
    ]

    ordinal_encoder = OrdinalEncoder(categories=categories)
    ordinal_encoded = ordinal_encoder.fit_transform(df[['尺寸', '等级']])

    df_ordinal = pd.DataFrame(
        ordinal_encoded,
        columns=['尺寸_序数', '等级_序数']
    )

    print("\n序数编码后的数据:")
    print(df_ordinal)
    print("\n注意: 序数编码保留了类别之间的顺序关系")


def missing_value_imputation():
    """
    演示不同的缺失值处理方法
    """
    print("\n" + "=" * 50)
    print("缺失值处理".center(50))
    print("=" * 50)

    # 创建带有缺失值的示例数据
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1, size=(100, 4))

    # 随机引入缺失值
    mask = np.random.random(size=data.shape) < 0.2
    data_with_nan = data.copy()
    data_with_nan[mask] = np.nan

    df = pd.DataFrame(
        data_with_nan,
        columns=['特征1', '特征2', '特征3', '特征4']
    )

    print("\n带有缺失值的原始数据:")
    print(f"形状: {df.shape}")
    print(f"缺失值数量: {df.isna().sum()}")
    print(df.head())

    # 1. 简单填充 (SimpleImputer)
    print("\n1. 简单填充 (SimpleImputer)")

    # 均值填充
    mean_imputer = SimpleImputer(strategy='mean')
    df_mean_imputed = pd.DataFrame(
        mean_imputer.fit_transform(df),
        columns=df.columns
    )

    # 中位数填充
    median_imputer = SimpleImputer(strategy='median')
    df_median_imputed = pd.DataFrame(
        median_imputer.fit_transform(df),
        columns=df.columns
    )

    # 众数填充
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df_mode_imputed = pd.DataFrame(
        mode_imputer.fit_transform(df),
        columns=df.columns
    )

    # 常数填充
    constant_imputer = SimpleImputer(strategy='constant', fill_value=0)
    df_constant_imputed = pd.DataFrame(
        constant_imputer.fit_transform(df),
        columns=df.columns
    )

    print("\n不同填充策略的比较:")
    print(f"原始数据均值: {df['特征1'].mean():.4f}")
    print(f"均值填充后的均值: {df_mean_imputed['特征1'].mean():.4f}")
    print(f"中位数填充后的均值: {df_median_imputed['特征1'].mean():.4f}")
    print(f"众数填充后的均值: {df_mode_imputed['特征1'].mean():.4f}")
    print(f"常数填充后的均值: {df_constant_imputed['特征1'].mean():.4f}")

    # 2. KNN填充 (KNNImputer)
    print("\n2. KNN填充 (KNNImputer)")
    print("使用K近邻算法估计缺失值")

    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn_imputed = pd.DataFrame(
        knn_imputer.fit_transform(df),
        columns=df.columns
    )

    print("\nKNN填充后的数据:")
    print(f"KNN填充后的均值: {df_knn_imputed['特征1'].mean():.4f}")

    # 3. 缺失值处理的最佳实践
    print("\n3. 缺失值处理的最佳实践:")
    print("- 了解缺失值的原因和模式")
    print("- 考虑缺失值是否包含信息（缺失不是随机的）")
    print("- 对于小数据集，考虑删除包含缺失值的行或列")
    print("- 对于大数据集，使用适当的填充方法")
    print("- 考虑添加'缺失指示器'特征，标记原始值是否缺失")
    print("- 在交叉验证中包含缺失值处理步骤，避免数据泄露")


def feature_engineering():
    """
    演示特征工程技术
    """
    print("\n" + "=" * 50)
    print("特征工程".center(50))
    print("=" * 50)

    # 加载波士顿房价数据集
    try:
        boston = datasets.load_boston()
        X = boston.data
        y = boston.target
        feature_names = boston.feature_names
    except:
        # 如果波士顿数据集不可用，使用加州房价数据集
        california = datasets.fetch_california_housing()
        X = california.data
        y = california.target
        feature_names = california.feature_names

    print(f"\n数据集形状: X={X.shape}, y={y.shape}")
    print(f"特征名称: {feature_names}")

    # 1. 多项式特征
    print("\n1. 多项式特征 (PolynomialFeatures)")
    print("创建原始特征的多项式组合")
    
    # 选择前2个特征用于演示
    X_subset = X[:, :2]
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_subset)
    
    print(f"原始特征形状: {X_subset.shape}")
    print(f"多项式特征形状: {X_poly.shape}")
    print(f"多项式特征名称: {poly.get_feature_names_out(['x1', 'x2'])}")
    
    # 2. 特征选择
    print("\n2. 特征选择")
    
    # 基于方差的特征选择
    print("\n2.1 基于方差的特征选择 (VarianceThreshold)")
    print("移除方差低于阈值的特征")
    
    selector = VarianceThreshold(threshold=0.1)
    X_var_selected = selector.fit_transform(X)
    
    print(f"原始特征数量: {X.shape[1]}")
    print(f"选择后的特征数量: {X_var_selected.shape[1]}")
    print(f"被选择的特征索引: {selector.get_support(indices=True)}")
    
    # 单变量特征选择
    print("\n2.2 单变量特征选择 (SelectKBest)")
    print("基于统计测试选择K个最佳特征")
    
    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(X, y > np.median(y))  # 将回归问题转换为分类问题
    
    print(f"选择后的特征数量: {X_new.shape[1]}")
    print(f"特征得分: {selector.scores_}")
    print(f"被选择的特征索引: {selector.get_support(indices=True)}")
    
    # 递归特征消除
    print("\n2.3 递归特征消除 (RFE)")
    print("通过递归地训练模型并移除最不重要的特征来选择特征")
    
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator=model, n_features_to_select=5)
    X_rfe = rfe.fit_transform(X, y > np.median(y))
    
    print(f"选择后的特征数量: {X_rfe.shape[1]}")
    print(f"特征排名 (1表示被选择): {rfe.ranking_}")
    print(f"被选择的特征索引: {rfe.get_support(indices=True)}")
    
    # 3. 主成分分析 (PCA)
    print("\n3. 主成分分析 (PCA)")
    print("将数据投影到主成分上以降维")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"原始特征数量: {X.shape[1]}")
    print(f"PCA后的特征数量: {X_pca.shape[1]}")
    print(f"解释方差比: {pca.explained_variance_ratio_}")
    print(f"累计解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 可视化PCA结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.colorbar(label='目标值')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.title('PCA降维结果')
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/pca_visualization.png")
    plt.close()
    
    print("\nPCA可视化已保存为 'images/pca_visualization.png'")


def pipeline_and_column_transformer():
    """
    演示Pipeline和ColumnTransformer的使用
    """
    print("\n" + "=" * 50)
    print("Pipeline和ColumnTransformer".center(50))
    print("=" * 50)
    
    # 创建混合数据集（数值和分类特征）
    np.random.seed(42)
    n_samples = 100
    
    # 数值特征
    numeric_features = np.random.normal(size=(n_samples, 3))
    
    # 分类特征
    categorical_features = np.random.choice(
        ['A', 'B', 'C'], size=(n_samples, 2)
    )
    
    # 目标变量
    y = np.random.randint(0, 2, size=n_samples)
    
    # 创建DataFrame
    X = pd.DataFrame(
        np.hstack([numeric_features, categorical_features]),
        columns=['数值1', '数值2', '数值3', '分类1', '分类2']
    )
    
    print("\n混合数据集示例:")
    print(X.head())
    print(f"\n目标变量分布:\n{pd.Series(y).value_counts()}")
    
    # 1. Pipeline
    print("\n1. Pipeline")
    print("将多个处理步骤链接在一起")
    
    # 创建一个简单的Pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # 应用到数值特征
    X_numeric = X[['数值1', '数值2', '数值3']].values
    X_numeric_transformed = numeric_pipeline.fit_transform(X_numeric)
    
    print("\nPipeline转换后的数值特征:")
    print(pd.DataFrame(
        X_numeric_transformed, 
        columns=['数值1_转换', '数值2_转换', '数值3_转换']
    ).head())
    
    # 2. ColumnTransformer
    print("\n2. ColumnTransformer")
    print("对不同列应用不同的转换")
    
    # 定义预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, ['数值1', '数值2', '数值3']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['分类1', '分类2'])
        ],
        remainder='drop'  # 丢弃未指定的列
    )
    
    # 创建完整的Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    full_pipeline.fit(X_train, y_train)
    
    # 评估模型
    score = full_pipeline.score(X_test, y_test)
    print(f"\n模型准确率: {score:.4f}")
    
    # 3. 获取Pipeline中的参数
    print("\n3. 获取Pipeline中的参数")
    print("使用双下划线语法访问Pipeline中的参数")
    
    print("\nLogisticRegression的系数:")
    classifier = full_pipeline.named_steps['classifier']
    print(classifier.coef_)
    
    # 4. Pipeline的优势
    print("\n4. Pipeline的优势:")
    print("- 代码更简洁、更易于理解")
    print("- 防止数据泄露（在交叉验证中正确应用转换）")
    print("- 简化模型部署（只需保存一个对象）")
    print("- 参数网格搜索更容易（可以优化预处理和模型参数）")


def main():
    """主函数，运行所有示例"""
    # 设置matplotlib显示中文
    set_matplotlib_chinese()
    feature_scaling()
    categorical_encoding()
    missing_value_imputation()
    feature_engineering()
    pipeline_and_column_transformer()


if __name__ == "__main__":
    main()