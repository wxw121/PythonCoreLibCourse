"""
Matplotlib高级教程

本模块涵盖Matplotlib的高级特性，包括：
1. 3D绘图
2. 动画
3. 自定义投影
4. 图像处理
5. 高级样式设置
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib_seaborn_tutorial import config

# 设置中文显示
config.setup_chinese_display()

def plot_3d_surface():
    """
    3D表面图示例
    """
    print("\n" + "=" * 50)
    print("3D表面图")
    print("=" * 50)

    # 创建数据
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D表面
    surface = ax.plot_surface(X, Y, Z, cmap='viridis',
                            linewidth=0, antialiased=True)

    # 添加颜色条
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # 设置标题和标签
    ax.set_title('3D表面图示例', fontsize=16)
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 调整视角
    ax.view_init(elev=30, azim=45)

    # 显示图形
    plt.show()

    print("\n3D表面图的关键点:")
    print("1. 使用add_subplot(projection='3d')创建3D坐标系")
    print("2. plot_surface()方法绘制3D表面")
    print("3. view_init()方法调整视角")
    print("4. 可以添加颜色映射和颜色条")

def plot_3d_scatter():
    """
    3D散点图示例
    """
    print("\n" + "=" * 50)
    print("3D散点图")
    print("=" * 50)

    # 创建随机数据
    n_points = 1000
    x = np.random.normal(0, 1, n_points)
    y = np.random.normal(0, 1, n_points)
    z = np.random.normal(0, 1, n_points)

    # 计算到原点的距离作为颜色映射
    colors = np.sqrt(x**2 + y**2 + z**2)

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D散点图
    scatter = ax.scatter(x, y, z, c=colors, cmap='viridis',
                        alpha=0.6, s=30)

    # 添加颜色条
    fig.colorbar(scatter, ax=ax, label='距离')

    # 设置标题和标签
    ax.set_title('3D散点图示例', fontsize=16)
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 显示图形
    plt.show()

    print("\n3D散点图的关键点:")
    print("1. 使用scatter()方法绘制3D散点")
    print("2. 可以通过颜色映射显示第四个维度的数据")
    print("3. alpha参数控制点的透明度")

def create_animation():
    """
    动画示例
    """
    print("\n" + "=" * 50)
    print("动画示例")
    print("=" * 50)

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 6))

    # 设置轴的范围
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # 创建一个点（将被动画化）
    point, = ax.plot([], [], 'ro')

    # 初始化函数
    def init():
        point.set_data([], [])
        return point,

    # 动画更新函数
    def animate(frame):
        # 计算点的新位置
        t = frame / 50.0
        x = np.cos(2 * np.pi * t)
        y = np.sin(2 * np.pi * t)
        point.set_data([x], [y])
        return point,

    # 创建动画
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=200, interval=20, blit=True)

    # 设置标题和网格
    ax.set_title('简单动画示例', fontsize=16)
    ax.grid(True)

    # 显示动画
    plt.show()

    print("\n动画的关键点:")
    print("1. 使用animation.FuncAnimation创建动画")
    print("2. 需要定义初始化函数和更新函数")
    print("3. frames参数控制帧数")
    print("4. interval参数控制帧间隔（毫秒）")

def custom_projection():
    """
    自定义投影示例
    """
    print("\n" + "=" * 50)
    print("自定义投影")
    print("=" * 50)

    # 创建极坐标数据
    r = np.linspace(0, 2, 100)
    theta = np.linspace(0, 6*np.pi, 100)
    r, theta = np.meshgrid(r, theta)

    # 计算笛卡尔坐标
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Z = np.sin(5*theta)

    # 创建图形
    fig = plt.figure(figsize=(15, 5))

    # 创建三个子图，使用不同的投影
    # 标准3D视图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title('标准3D视图')

    # 从顶部看
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis')
    ax2.view_init(90, 0)  # 俯视图
    ax2.set_title('顶视图')

    # 从侧面看
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, Z, cmap='viridis')
    ax3.view_init(0, 90)  # 侧视图
    ax3.set_title('侧视图')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    print("\n自定义投影的关键点:")
    print("1. 可以使用不同的投影方式查看3D数据")
    print("2. view_init()方法可以调整视角")
    print("3. 多个视角可以帮助理解3D数据结构")

def image_processing():
    """
    图像处理示例
    """
    print("\n" + "=" * 50)
    print("图像处理")
    print("=" * 50)

    # 创建示例图像数据
    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 原始图像
    im1 = ax1.imshow(Z, cmap='viridis')
    ax1.set_title('原始图像')
    fig.colorbar(im1, ax=ax1)

    # 添加噪声
    Z_noisy = Z + np.random.normal(0, 0.2, Z.shape)
    im2 = ax2.imshow(Z_noisy, cmap='viridis')
    ax2.set_title('添加噪声')
    fig.colorbar(im2, ax=ax2)

    # 等高线图
    im3 = ax3.contour(X, Y, Z, levels=15, cmap='viridis')
    ax3.set_title('等高线图')
    fig.colorbar(im3, ax=ax3)

    # 填充等高线图
    im4 = ax4.contourf(X, Y, Z, levels=15, cmap='viridis')
    ax4.set_title('填充等高线图')
    fig.colorbar(im4, ax=ax4)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    print("\n图像处理的关键点:")
    print("1. imshow()用于显示图像数据")
    print("2. contour()创建等高线图")
    print("3. contourf()创建填充等高线图")
    print("4. 可以添加颜色条显示数值范围")

def advanced_styling():
    """
    高级样式设置示例
    """
    print("\n" + "=" * 50)
    print("高级样式设置")
    print("=" * 50)
    
    # 创建示例数据
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 2))

    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 自定义颜色映射
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    points = ax1.scatter(data[:, 0], data[:, 1], c=data[:, 0],
                        cmap='coolwarm', s=100)
    ax1.set_title('自定义颜色映射')
    fig.colorbar(points, ax=ax1)

    # 2. 自定义标记和线型
    ax2.plot(x, y, 'g--', marker='o', markersize=8,
             markerfacecolor='white', markeredgecolor='green',
             markeredgewidth=2, label='自定义线型')
    ax2.set_title('自定义标记和线型')
    ax2.legend()

    # 3. 添加自定义图形
    # 创建一些补丁
    patches = []
    circle = Circle((0.5, 0.5), 0.2, alpha=0.5)
    rect = Rectangle((0.2, 0.2), 0.3, 0.4, alpha=0.5)
    polygon = Polygon([[0.6, 0.2], [0.8, 0.3], [0.7, 0.6]], alpha=0.5)

    patches.append(circle)
    patches.append(rect)
    patches.append(polygon)

    # 添加补丁集合
    p = PatchCollection(patches, cmap='viridis', alpha=0.5)
    ax3.add_collection(p)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('自定义图形')

    # 4. 自定义网格和背景
    ax4.plot(x, y, 'r-', linewidth=2)
    ax4.set_facecolor('#f0f0f0')  # 设置背景色
    ax4.grid(True, linestyle='--', alpha=0.7, color='white', linewidth=2)
    ax4.set_title('自定义网格和背景')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    print("\n高级样式设置的关键点:")
    print("1. 可以自定义颜色映射和标记样式")
    print("2. 可以添加自定义图形（补丁）")
    print("3. 可以自定义网格和背景")
    print("4. 可以调整透明度和其他视觉效果")

def run_example():
    """
    运行所有示例
    """
    # 3D绘图
    plot_3d_surface()
    plot_3d_scatter()

    # 动画
    create_animation()

    # 自定义投影
    custom_projection()

    # 图像处理
    image_processing()

    # 高级样式设置
    advanced_styling()

    print("\n" + "=" * 50)
    print("Matplotlib高级教程完成！")
    print("=" * 50)

if __name__ == "__main__":
    run_example()
