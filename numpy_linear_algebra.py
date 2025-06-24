"""
NumPy线性代数教程
本教程介绍NumPy库的线性代数功能，包括矩阵操作、矩阵分解、特征值计算等内容。
"""

import numpy as np

def matrix_operations():
    """
    演示NumPy的基本矩阵操作
    包括：矩阵创建、转置、求逆等
    """
    print("="*50)
    print("NumPy矩阵操作")
    print("="*50)

    # 创建矩阵
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    B = np.array([[9, 8, 7],
                  [6, 5, 4],
                  [3, 2, 1]])

    print("\n1. 创建的矩阵:")
    print(f"矩阵A:\n{A}")
    print(f"矩阵B:\n{B}")

    # 矩阵转置
    print("\n2. 矩阵转置:")
    A_T = A.T
    print(f"A的转置:\n{A_T}")

    # 矩阵乘法
    print("\n3. 矩阵乘法:")
    # 使用np.dot或@运算符进行矩阵乘法
    C1 = np.dot(A, B)
    C2 = A @ B  # Python 3.5+支持的矩阵乘法运算符

    print(f"A @ B (使用np.dot):\n{C1}")
    print(f"A @ B (使用@运算符):\n{C2}")

    # 元素级乘法
    print("\n4. 元素级乘法:")
    element_wise = A * B
    print(f"A * B (元素级):\n{element_wise}")

    # 矩阵求逆
    # 创建一个可逆矩阵
    D = np.array([[1, 2], [3, 4]])
    print("\n5. 矩阵求逆:")
    print(f"矩阵D:\n{D}")

    try:
        D_inv = np.linalg.inv(D)
        print(f"D的逆矩阵:\n{D_inv}")

        # 验证求逆是否正确
        print("\n验证D * D^(-1)是否接近单位矩阵:")
        print(np.dot(D, D_inv))
    except np.linalg.LinAlgError:
        print("矩阵不可逆")

def matrix_decomposition():
    """
    演示NumPy的矩阵分解功能
    包括：LU分解、QR分解、奇异值分解(SVD)、特征值分解
    """
    print("\n"+"="*50)
    print("NumPy矩阵分解")
    print("="*50)

    # 创建一个示例矩阵
    A = np.array([[1, 2], [3, 4]])
    print(f"\n示例矩阵A:\n{A}")

    # LU分解
    from scipy.linalg import lu
    print("\n1. LU分解:")
    try:
        P, L, U = lu(A)
        print(f"P (置换矩阵):\n{P}")
        print(f"L (下三角矩阵):\n{L}")
        print(f"U (上三角矩阵):\n{U}")
        print("\n验证P @ L @ U = A:")
        print(P @ L @ U)  # 在Python 3.5+中使用@运算符
    except ImportError:
        print("需要安装SciPy库来执行LU分解")

    # QR分解
    print("\n2. QR分解:")
    Q, R = np.linalg.qr(A)
    print(f"Q (正交矩阵):\n{Q}")
    print(f"R (上三角矩阵):\n{R}")
    print("\n验证Q @ R = A:")
    print(Q @ R)

    # 奇异值分解(SVD)
    print("\n3. 奇异值分解(SVD):")
    U, S, VT = np.linalg.svd(A)
    print(f"U (左奇异向量):\n{U}")
    print(f"S (奇异值):\n{S}")
    print(f"V^T (右奇异向量的转置):\n{VT}")

    # 重构原始矩阵
    # 需要将S转换为对角矩阵
    S_diag = np.diag(S)
    print("\n验证U @ S @ V^T = A:")
    # 由于S是一维数组，我们需要将其转换为对角矩阵
    print(U @ S_diag @ VT)

    # 特征值分解
    print("\n4. 特征值分解:")
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"特征值:\n{eigenvalues}")
    print(f"特征向量 (按列排列):\n{eigenvectors}")

    # 验证Av = λv
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lambda_v = eigenvalues[i]
        Av = A @ v
        lambda_times_v = lambda_v * v
        print(f"\n验证第{i+1}个特征向量:")
        print(f"A @ v = {Av}")
        print(f"λ * v = {lambda_times_v}")
        print(f"差值的范数: {np.linalg.norm(Av - lambda_times_v)}")

def solving_linear_systems():
    """
    演示如何使用NumPy解线性方程组
    """
    print("\n"+"="*50)
    print("解线性方程组")
    print("="*50)

    # 创建系数矩阵和常数向量
    # 解方程组: 2x + y = 8
    #          3x + 5y = 19
    A = np.array([[2, 1], [3, 5]])
    b = np.array([8, 19])

    print("\n1. 线性方程组:")
    print("2x + y = 8")
    print("3x + 5y = 19")

    # 使用np.linalg.solve解方程组
    print("\n2. 使用np.linalg.solve:")
    x = np.linalg.solve(A, b)
    print(f"解: x = {x[0]}, y = {x[1]}")

    # 验证解是否正确
    print("\n验证解:")
    print(f"A @ x = {A @ x}")
    print(f"b = {b}")

    # 使用矩阵求逆解方程组
    print("\n3. 使用矩阵求逆:")
    x_inv = np.linalg.inv(A) @ b
    print(f"解: x = {x_inv[0]}, y = {x_inv[1]}")

    # 最小二乘解（对于过定方程组）
    print("\n4. 最小二乘解 (对于过定方程组):")
    # 创建一个过定方程组 (更多方程than未知数)
    A_over = np.array([[2, 1], [3, 5], [1, 1]])
    b_over = np.array([8, 19, 5])

    print("方程组:")
    print("2x + y = 8")
    print("3x + 5y = 19")
    print("x + y = 5")

    # 使用最小二乘法求解
    x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
    print(f"最小二乘解: x = {x_lstsq[0]}, y = {x_lstsq[1]}")
    print(f"残差: {residuals}")

def matrix_norms():
    """
    演示NumPy中的矩阵范数计算
    """
    print("\n"+"="*50)
    print("矩阵范数")
    print("="*50)

    # 创建示例矩阵
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"\n示例矩阵A:\n{A}")

    # Frobenius范数
    print("\n1. Frobenius范数:")
    frob_norm = np.linalg.norm(A, 'fro')
    print(f"||A||_F = {frob_norm}")
    print("(Frobenius范数是矩阵所有元素平方和的平方根)")

    # 1-范数 (最大列和)
    print("\n2. 1-范数 (最大列和):")
    norm_1 = np.linalg.norm(A, 1)
    print(f"||A||_1 = {norm_1}")
    print("(1-范数是矩阵各列绝对值之和的最大值)")

    # 无穷范数 (最大行和)
    print("\n3. 无穷范数 (最大行和):")
    norm_inf = np.linalg.norm(A, np.inf)
    print(f"||A||_∞ = {norm_inf}")
    print("(无穷范数是矩阵各行绝对值之和的最大值)")

    # 2-范数 (最大奇异值)
    print("\n4. 2-范数 (最大奇异值):")
    norm_2 = np.linalg.norm(A, 2)
    print(f"||A||_2 = {norm_2}")
    print("(2-范数是矩阵最大奇异值)")

    # 核范数 (奇异值之和)
    print("\n5. 核范数 (奇异值之和):")
    # 计算奇异值
    s = np.linalg.svd(A, compute_uv=False)
    nuclear_norm = sum(s)
    print(f"||A||_* = {nuclear_norm}")
    print("(核范数是矩阵所有奇异值的和)")

def main():
    """
    主函数，按顺序运行所有示例
    """
    matrix_operations()
    matrix_decomposition()
    solving_linear_systems()
    matrix_norms()

if __name__ == "__main__":
    main()
