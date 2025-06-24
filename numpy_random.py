"""
NumPy随机数生成教程
本教程介绍NumPy库的随机数生成功能，包括各种概率分布、随机抽样、随机排列等内容。
"""

import numpy as np
import matplotlib.pyplot as plt

def random_number_basics():
    """
    演示NumPy随机数生成的基础知识
    包括：随机数生成器的创建、种子设置、基本随机数生成
    """
    print("="*50)
    print("NumPy随机数基础")
    print("="*50)

    # 1. 创建随机数生成器
    print("\n1. 创建随机数生成器:")
    # 在NumPy 1.17.0之后，推荐使用Generator而不是旧的RandomState
    rng = np.random.default_rng(seed=42)  # 创建一个带有种子的生成器
    print("已创建随机数生成器，种子值为42")

    # 2. 生成随机浮点数
    print("\n2. 生成随机浮点数:")
    # 生成0到1之间的随机浮点数
    random_float = rng.random()
    print(f"单个随机浮点数: {random_float}")

    # 生成一个随机浮点数数组
    random_array = rng.random(5)
    print(f"随机浮点数数组: {random_array}")

    # 生成一个2D随机浮点数数组
    random_2d = rng.random((2, 3))
    print(f"2D随机浮点数数组:\n{random_2d}")

    # 3. 生成指定范围内的随机整数
    print("\n3. 生成随机整数:")
    # 生成[0, 10)范围内的随机整数
    random_int = rng.integers(0, 10)
    print(f"单个随机整数 [0, 10): {random_int}")

    # 生成[0, 100)范围内的5个随机整数
    random_ints = rng.integers(0, 100, size=5)
    print(f"随机整数数组 [0, 100): {random_ints}")

    # 4. 设置随机种子
    print("\n4. 设置随机种子的重要性:")
    # 创建两个相同种子的生成器
    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)

    print(f"生成器1的随机数: {rng1.random(3)}")
    print(f"生成器2的随机数: {rng2.random(3)}")
    print("注意：相同的种子产生相同的随机数序列，这对于结果复现很重要")

    # 创建两个不同种子的生成器
    rng3 = np.random.default_rng(seed=42)
    rng4 = np.random.default_rng(seed=43)

    print(f"生成器3的随机数: {rng3.random(3)}")
    print(f"生成器4的随机数: {rng4.random(3)}")
    print("注意：不同的种子产生不同的随机数序列")

def probability_distributions():
    """
    演示NumPy中的各种概率分布
    包括：均匀分布、正态分布、二项分布、泊松分布等
    """
    print("\n"+"="*50)
    print("NumPy概率分布")
    print("="*50)

    # 创建随机数生成器
    rng = np.random.default_rng(seed=42)

    # 1. 均匀分布
    print("\n1. 均匀分布:")
    # 生成[0, 1)范围内的均匀分布随机数
    uniform_samples = rng.random(1000)
    print(f"均匀分布样本的均值: {uniform_samples.mean():.4f}")
    print(f"均匀分布样本的标准差: {uniform_samples.std():.4f}")
    print("理论均值: 0.5, 理论标准差: 0.2887")

    # 生成[5, 15)范围内的均匀分布随机数
    uniform_range = rng.uniform(5, 15, 1000)
    print(f"范围[5, 15)内均匀分布样本的均值: {uniform_range.mean():.4f}")
    print(f"范围[5, 15)内均匀分布样本的标准差: {uniform_range.std():.4f}")

    # 2. 正态分布（高斯分布）
    print("\n2. 正态分布:")
    # 生成均值为0，标准差为1的标准正态分布随机数
    normal_samples = rng.normal(0, 1, 1000)
    print(f"标准正态分布样本的均值: {normal_samples.mean():.4f}")
    print(f"标准正态分布样本的标准差: {normal_samples.std():.4f}")
    print("理论均值: 0, 理论标准差: 1")

    # 生成均值为50，标准差为10的正态分布随机数
    normal_custom = rng.normal(50, 10, 1000)
    print(f"自定义正态分布样本的均值: {normal_custom.mean():.4f}")
    print(f"自定义正态分布样本的标准差: {normal_custom.std():.4f}")

    # 3. 二项分布
    print("\n3. 二项分布:")
    # 模拟抛10次硬币，每次成功概率为0.5
    binomial_samples = rng.binomial(n=10, p=0.5, size=1000)
    print(f"二项分布样本的均值: {binomial_samples.mean():.4f}")
    print(f"二项分布样本的标准差: {binomial_samples.std():.4f}")
    print("理论均值: 5, 理论标准差: 1.5811")

    # 4. 泊松分布
    print("\n4. 泊松分布:")
    # 生成均值为5的泊松分布随机数
    poisson_samples = rng.poisson(lam=5, size=1000)
    print(f"泊松分布样本的均值: {poisson_samples.mean():.4f}")
    print(f"泊松分布样本的标准差: {poisson_samples.std():.4f}")
    print("理论均值: 5, 理论标准差: 2.2361")

    # 5. 指数分布
    print("\n5. 指数分布:")
    # 生成尺度参数为1的指数分布随机数
    exponential_samples = rng.exponential(scale=1.0, size=1000)
    print(f"指数分布样本的均值: {exponential_samples.mean():.4f}")
    print(f"指数分布样本的标准差: {exponential_samples.std():.4f}")
    print("理论均值: 1, 理论标准差: 1")

def random_sampling():
    """
    演示NumPy的随机抽样功能
    包括：简单随机抽样、加权随机抽样
    """
    print("\n"+"="*50)
    print("NumPy随机抽样")
    print("="*50)

    # 创建随机数生成器
    rng = np.random.default_rng(seed=42)

    # 1. 简单随机抽样
    print("\n1. 简单随机抽样:")
    # 从一个数组中随机抽取元素
    population = np.arange(10, 30)  # 创建一个包含10到29的数组
    print(f"总体: {population}")

    # 不放回抽样
    sample_without_replacement = rng.choice(population, size=5, replace=False)
    print(f"不放回抽样: {sample_without_replacement}")

    # 有放回抽样
    sample_with_replacement = rng.choice(population, size=5, replace=True)
    print(f"有放回抽样: {sample_with_replacement}")

    # 2. 加权随机抽样
    print("\n2. 加权随机抽样:")
    # 创建一个数组和对应的权重
    items = ['A', 'B', 'C', 'D', 'E']
    weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # 权重之和为1
    print(f"项目: {items}")
    print(f"权重: {weights}")

    # 根据权重进行抽样
    weighted_sample = rng.choice(items, size=10, p=weights)
    print(f"加权抽样结果: {weighted_sample}")

    # 统计每个元素出现的次数
    unique, counts = np.unique(weighted_sample, return_counts=True)
    print("抽样频率:")
    for item, count in zip(unique, counts):
        print(f"{item}: {count/10:.2f}")

def random_permutations():
    """
    演示NumPy的随机排列和洗牌功能
    """
    print("\n"+"="*50)
    print("NumPy随机排列和洗牌")
    print("="*50)

    # 创建随机数生成器
    rng = np.random.default_rng(seed=42)

    # 1. 随机排列
    print("\n1. 随机排列:")
    # 创建一个有序数组
    arr = np.arange(10)
    print(f"原始数组: {arr}")

    # 生成随机排列
    permutation = rng.permutation(arr)
    print(f"随机排列: {permutation}")
    print("注意：原始数组保持不变")
    print(f"原始数组: {arr}")

    # 2. 随机洗牌
    print("\n2. 随机洗牌:")
    # 创建一个有序数组
    arr2 = np.arange(10)
    print(f"原始数组: {arr2}")

    # 对数组进行洗牌（就地修改）
    rng.shuffle(arr2)
    print(f"洗牌后的数组: {arr2}")
    print("注意：原始数组被修改")

def visualization_examples():
    """
    使用matplotlib可视化NumPy生成的随机分布
    """
    print("\n"+"="*50)
    print("随机分布可视化")
    print("="*50)
    print("\n注意：此函数会生成图形，但在控制台环境中可能不会显示。")
    print("如果在支持图形显示的环境中运行，将会看到分布的直方图。")

    # 创建随机数生成器
    rng = np.random.default_rng(seed=42)

    # 创建一个图形，包含2行2列的子图
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 1. 均匀分布
    uniform_samples = rng.random(1000)
    axs[0, 0].hist(uniform_samples, bins=30, alpha=0.7)
    axs[0, 0].set_title('均匀分布 Uniform(0, 1)')
    axs[0, 0].set_xlabel('值')
    axs[0, 0].set_ylabel('频率')

    # 2. 正态分布
    normal_samples = rng.normal(0, 1, 1000)
    axs[0, 1].hist(normal_samples, bins=30, alpha=0.7)
    axs[0, 1].set_title('正态分布 Normal(0, 1)')
    axs[0, 1].set_xlabel('值')
    axs[0, 1].set_ylabel('频率')

    # 3. 二项分布
    binomial_samples = rng.binomial(n=10, p=0.5, size=1000)
    axs[1, 0].hist(binomial_samples, bins=range(0, 12), alpha=0.7)
    axs[1, 0].set_title('二项分布 Binomial(10, 0.5)')
    axs[1, 0].set_xlabel('值')
    axs[1, 0].set_ylabel('频率')

    # 4. 泊松分布
    poisson_samples = rng.poisson(lam=5, size=1000)
    axs[1, 1].hist(poisson_samples, bins=range(0, 15), alpha=0.7)
    axs[1, 1].set_title('泊松分布 Poisson(5)')
    axs[1, 1].set_xlabel('值')
    axs[1, 1].set_ylabel('频率')

    # 调整布局
    plt.tight_layout()

    # 保存图形到文件
    plt.savefig('numpy_distributions.png')
    print("\n已将分布图保存为 'numpy_distributions.png'")

def main():
    """
    主函数，按顺序运行所有示例
    """
    random_number_basics()
    probability_distributions()
    random_sampling()
    random_permutations()

    # 如果matplotlib可用，则运行可视化示例
    try:
        visualization_examples()
    except Exception as e:
        print(f"\n无法运行可视化示例: {e}")