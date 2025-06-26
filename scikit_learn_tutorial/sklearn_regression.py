"""
Scikit-learn 回归算法

本模块介绍scikit-learn中的回归算法，包括线性回归、岭回归、Lasso回归和多项式回归等。
回归是监督学习的一种，目标是预测连续的数值。
"""

import os
# 导入matplotlib配置（必须在任何其他matplotlib导入之前）
from matplotlib_config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)

from scikit_learn_tutorial.config import set_matplotlib_chinese


def load_data():
    """
    加载和准备数据集
    """
    # 加载波士顿房价数据集（如果可用）
    try:
        boston = datasets.load_boston()
        X = boston.data
        y = boston.target
        feature_names = boston.feature_names
        dataset_name = "Boston Housing"
    except:
        # 如果波士顿数据集不可用，使用加州房价数据集
        california = datasets.fetch_california_housing()
        X = california.data
        y = california.target
        feature_names = california.feature_names
        dataset_name = "California Housing"
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            feature_names, dataset_name)


def evaluate_model(model, X_test, y_test, model_name):
    """
    评估回归模型并打印结果
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred)
    
    # 打印评估结果
    print(f"\n{model_name} 评估结果:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"R²分数: {r2:.4f}")
    print(f"解释方差分数: {ev:.4f}")
    
    # 可视化预测值与真实值的对比
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{model_name} - 预测值 vs 真实值')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{model_name.lower().replace(' ', '_')}_predictions.png")
    plt.close()
    
    # 可视化残差
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title(f'{model_name} - 残差图')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{model_name.lower().replace(' ', '_')}_residuals.png")
    plt.close()

    return mse, rmse, mae, r2, ev


def linear_regression_example():
    """
    线性回归示例
    """
    print("=" * 50)
    print("线性回归".center(50))
    print("=" * 50)

    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, dataset_name = load_data()

    print(f"\n使用{dataset_name}数据集")
    print("\n线性回归是最基本的回归算法，假设目标变量与特征之间存在线性关系。")

    # 创建并训练线性回归模型
    print("\n训练线性回归模型...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 查看模型系数
    print("\n模型系数:")
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': model.coef_
    })
    print(coef_df)

    # 可视化特征系数
    plt.figure(figsize=(12, 8))
    coef_df = coef_df.reindex(coef_df.系数.abs().sort_values(ascending=True).index)
    colors = ['red' if c < 0 else 'blue' for c in coef_df['系数']]
    plt.barh(coef_df['特征'], coef_df['系数'], color=colors)
    plt.xlabel('系数值')
    plt.title('线性回归 - 特征系数')
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/linear_regression_coefficients.png")
    plt.close()

    print("\n特征系数图已保存为 'linear_regression_coefficients.png'")

    # 评估模型
    evaluate_model(model, X_test, y_test, "线性回归")

    # 线性回归的优缺点
    print("\n线性回归的优点:")
    print("- 简单、可解释性强")
    print("- 训练速度快")
    print("- 预测速度快")
    print("- 可以通过系数了解特征重要性")

    print("\n线性回归的缺点:")
    print("- 假设特征和目标变量之间是线性关系")
    print("- 对异常值敏感")
    print("- 特征之间不能有高度相关性")
    print("- 无法处理非线性关系")


def ridge_regression_example():
    """
    岭回归示例
    """
    print("\n" + "=" * 50)
    print("岭回归".center(50))
    print("=" * 50)

    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, dataset_name = load_data()

    print(f"\n使用{dataset_name}数据集")
    print("\n岭回归是线性回归的正则化版本，通过L2正则化来防止过拟合。")

    # 创建并训练岭回归模型
    print("\n训练岭回归模型...")

    # 使用不同的正则化强度
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_rmse = float('inf')
    best_model = None
    best_alpha = None

    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)

        # 在测试集上评估
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"alpha={alpha}: RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_alpha = alpha

    print(f"\n最佳正则化参数 alpha={best_alpha}, RMSE={best_rmse:.4f}")

    # 比较不同alpha值下的系数
    plt.figure(figsize=(12, 8))
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        plt.plot(model.coef_, 'o-', label=f'alpha={alpha}')
    plt.xlabel('特征索引')
    plt.ylabel('系数值')
    plt.title('岭回归 - 不同alpha值下的系数')
    plt.legend()
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/ridge_regression_coefficients.png")
    plt.close()

    print("\n不同alpha值下的系数图已保存为 'ridge_regression_coefficients.png'")

    # 评估最佳模型
    evaluate_model(best_model, X_test, y_test, "岭回归")

    # 岭回归的优缺点
    print("\n岭回归的优点:")
    print("- 可以处理特征之间的多重共线性")
    print("- 防止过拟合")
    print("- 可以处理特征数量大于样本数量的情况")
    print("- 保留所有特征")

    print("\n岭回归的缺点:")
    print("- 需要调整正则化参数")
    print("- 不进行特征选择（所有特征都保留）")
    print("- 仍然假设线性关系")


def lasso_regression_example():
    """
    Lasso回归示例
    """
    print("\n" + "=" * 50)
    print("Lasso回归".center(50))
    print("=" * 50)

    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, dataset_name = load_data()

    print(f"\n使用{dataset_name}数据集")
    print("\nLasso回归使用L1正则化，可以将不重要的特征系数压缩为零，实现特征选择。")

    # 创建并训练Lasso回归模型
    print("\n训练Lasso回归模型...")

    # 使用不同的正则化强度
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]
    best_rmse = float('inf')
    best_model = None
    best_alpha = None

    for alpha in alphas:
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)

        # 在测试集上评估
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # 计算非零系数的数量
        n_nonzero = np.sum(model.coef_ != 0)
        print(f"alpha={alpha}: RMSE={rmse:.4f}, 非零系数数量={n_nonzero}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_alpha = alpha

    print(f"\n最佳正则化参数 alpha={best_alpha}, RMSE={best_rmse:.4f}")

    # 比较不同alpha值下的系数
    plt.figure(figsize=(12, 8))
    for alpha in alphas:
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        plt.plot(model.coef_, 'o-', label=f'alpha={alpha}')
    plt.xlabel('特征索引')
    plt.ylabel('系数值')
    plt.title('Lasso回归 - 不同alpha值下的系数')
    plt.legend()
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/lasso_regression_coefficients.png")
    plt.close()

    print("\n不同alpha值下的系数图已保存为 'lasso_regression_coefficients.png'")

    # 评估最佳模型
    evaluate_model(best_model, X_test, y_test, "Lasso回归")

    # 查看选择的特征
    print("\n特征选择结果:")
    selected_features = pd.DataFrame({
        '特征': feature_names,
        '系数': best_model.coef_
    })
    selected_features = selected_features[selected_features['系数'] != 0]
    print("\n非零系数的特征:")
    print(selected_features)

    # Lasso回归的优缺点
    print("\nLasso回归的优点:")
    print("- 可以进行特征选择")
    print("- 产生稀疏解（部分系数为零）")
    print("- 防止过拟合")
    print("- 模型更简单，更容易解释")

    print("\nLasso回归的缺点:")
    print("- 需要调整正则化参数")
    print("- 在特征高度相关时可能表现不稳定")
    print("- 仍然假设线性关系")


def polynomial_regression_example():
    """
    多项式回归示例
    """
    print("\n" + "=" * 50)
    print("多项式回归".center(50))
    print("=" * 50)

    # 创建一个简单的非线性数据集用于演示
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)
    
    print("\n多项式回归通过添加多项式特征来捕捉非线性关系。")
    print("它结合了特征工程和线性回归。")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建并训练多项式回归模型
    print("\n训练多项式回归模型...")
    
    # 使用不同的多项式次数
    degrees = [1, 2, 3, 5, 10]
    models = {}
    rmse_scores = {}
    
    plt.figure(figsize=(14, 10))
    
    # 创建测试数据点用于可视化
    X_test_sorted = np.sort(X_test, axis=0)
    
    for i, degree in enumerate(degrees):
        # 创建多项式特征
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        
        # 创建Pipeline
        model = Pipeline([
            ("polynomial_features", polynomial_features),
            ("linear_regression", LinearRegression())
        ])
        
        # 训练模型
        model.fit(X_train, y_train)
        models[degree] = model
        
        # 预测
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores[degree] = rmse
        
        print(f"多项式次数={degree}: RMSE={rmse:.4f}")
        
        # 可视化
        plt.subplot(2, 3, i + 1)
        plt.scatter(X_train, y_train, color='blue', label='训练数据', alpha=0.5)
        plt.scatter(X_test, y_test, color='green', label='测试数据', alpha=0.5)
        
        # 预测曲线
        y_pred_line = model.predict(X_test_sorted)
        plt.plot(X_test_sorted, y_pred_line, color='red', label='预测')
        
        plt.title(f'多项式次数 = {degree}, RMSE = {rmse:.4f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
    
    plt.tight_layout()
    # 确保目录存在
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/polynomial_regression_comparison.png")
    plt.close()
    
    print("\n不同多项式次数的比较图已保存为 'polynomial_regression_comparison.png'")
    
    # 找出最佳模型
    best_degree = min(rmse_scores, key=rmse_scores.get)
    best_model = models[best_degree]
    best_rmse = rmse_scores[best_degree]
    
    print(f"\n最佳多项式次数 = {best_degree}, RMSE = {best_rmse:.4f}")
    
    # 多项式回归的优缺点
    print("\n多项式回归的优点:")
    print("- 可以捕捉非线性关系")
    print("- 基于线性回归，易于理解")
    print("- 可以通过调整多项式次数来控制模型复杂度")
    
    print("\n多项式回归的缺点:")
    print("- 高次多项式容易过拟合")
    print("- 需要选择合适的多项式次数")
    print("- 在高维数据上可能不实用（特征数量会迅速增加）")
    print("- 对异常值敏感")


def elastic_net_example():
    """
    弹性网络回归示例
    """
    print("\n" + "=" * 50)
    print("弹性网络回归".center(50))
    print("=" * 50)
    
    # 加载数据
    X_train, X_test, y_train, y_test, feature_names, dataset_name = load_data()
    
    print(f"\n使用{dataset_name}数据集")
    print("\n弹性网络回归结合了L1和L2正则化的优点，是Lasso和岭回归的混合。")
    
    # 创建并训练弹性网络模型
    print("\n训练弹性网络模型...")
    
    # 使用不同的参数组合
    alphas = [0.01, 0.1, 1.0]
    l1_ratios = [0.1, 0.5, 0.9]
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            model.fit(X_train, y_train)
            
            # 在测试集上评估
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # 计算非零系数的数量
            n_nonzero = np.sum(model.coef_ != 0)
            print(f"alpha={alpha}, l1_ratio={l1_ratio}: RMSE={rmse:.4f}, 非零系数数量={n_nonzero}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = (alpha, l1_ratio)
    
    print(f"\n最佳参数 alpha={best_params[0]}, l1_ratio={best_params[1]}, RMSE={best_rmse:.4f}")
    
    # 评估最佳模型
    evaluate_model(best_model, X_test, y_test, "弹性网络回归")
    
    # 弹性网络的优缺点
    print("\n弹性网络回归的优点:")
    print("- 结合了Lasso和岭回归的优点")
    print("- 在特征高度相关时比Lasso更稳定")
    print("- 可以进行特征选择")
    print("- 可以处理特征数量大于样本数量的情况")
    
    print("\n弹性网络回归的缺点:")
    print("- 需要调整两个参数（alpha和l1_ratio）")
    print("- 计算成本比单纯的Lasso或岭回归高")
    print("- 仍然假设线性关系")


def main():
    """主函数，运行所有示例"""
    # 设置matplotlib显示中文
    set_matplotlib_chinese()
    linear_regression_example()
    ridge_regression_example()
    lasso_regression_example()
    polynomial_regression_example()
    elastic_net_example()


if __name__ == "__main__":
    main()