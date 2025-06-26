"""
Scikit-learn 教程主入口

这是scikit-learn教程的主入口点，提供交互式界面让用户选择要运行的教程模块。
"""

import os
import sys
from importlib import import_module


def clear_screen():
    """清除控制台屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印教程标题"""
    print("\n" + "=" * 60)
    print("Scikit-learn 机器学习教程".center(60))
    print("=" * 60)


def print_menu():
    """打印主菜单"""
    print("\n请选择要运行的教程模块：")
    print("\n1. 基础知识 (sklearn_basics.py)")
    print("   - scikit-learn基本概念")
    print("   - 数据集加载和操作")
    print("   - 模型训练和预测的基本流程")

    print("\n2. 数据预处理 (sklearn_preprocessing.py)")
    print("   - 特征缩放和编码")
    print("   - 缺失值处理")
    print("   - 特征工程和选择")

    print("\n3. 分类算法 (sklearn_classification.py)")
    print("   - 逻辑回归")
    print("   - 决策树")
    print("   - 随机森林")
    print("   - SVM和KNN")

    print("\n4. 回归算法 (sklearn_regression.py)")
    print("   - 线性回归")
    print("   - 岭回归和Lasso")
    print("   - 多项式回归")

    print("\n5. 聚类算法 (sklearn_clustering.py)")
    print("   - K-means聚类")
    print("   - 层次聚类")
    print("   - DBSCAN")

    print("\n6. 模型评估 (sklearn_model_evaluation.py)")
    print("   - 交叉验证")
    print("   - 网格搜索")
    print("   - 评估指标")

    print("\n7. 实际应用案例 (sklearn_applications.py)")
    print("   - 文本分类")
    print("   - 图像识别")
    print("   - 推荐系统")

    print("\n8. 高级主题 (sklearn_advanced.py)")
    print("   - 模型集成")
    print("   - 特征选择")
    print("   - 异常检测")

    print("\n0. 退出教程")

    print("\n" + "=" * 60)


def run_module(module_name):
    """
    运行指定的教程模块

    Args:
        module_name: 要运行的模块名称
    """
    try:
        # 动态导入并运行模块
        module = import_module(module_name)
        clear_screen()
        print(f"\n运行 {module_name} ...")
        print("=" * 60 + "\n")
        module.main()

    except ImportError:
        print(f"\n错误：找不到模块 {module_name}")
        print("该模块可能尚未实现或名称错误。")
    except Exception as e:
        print(f"\n运行模块时出错：{str(e)}")

    input("\n按Enter键返回主菜单...")


def main():
    """主函数，运行教程界面"""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("\n请输入选项编号 (0-8): ").strip()

        if choice == '0':
            print("\n感谢使用Scikit-learn教程！再见！")
            break

        module_map = {
            '1': 'sklearn_basics',
            '2': 'sklearn_preprocessing',
            '3': 'sklearn_classification',
            '4': 'sklearn_regression',
            '5': 'sklearn_clustering',
            '6': 'sklearn_model_evaluation',
            '7': 'sklearn_applications',
            '8': 'sklearn_advanced'
        }

        if choice in module_map:
            run_module(module_map[choice])
        else:
            print("\n无效的选项，请重试！")
            input("\n按Enter键继续...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。感谢使用！")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序发生错误：{str(e)}")
        sys.exit(1)
