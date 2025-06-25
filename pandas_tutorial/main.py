"""
Pandas教程主入口
本模块作为Pandas教程的主入口，提供菜单选择不同的教程模块。
"""

import os
import sys
import importlib

def clear_screen():
    """
    清除控制台屏幕
    'nt' 表示 Windows，'posix' 表示 Linux/macOS
    如果是 Windows（os.name == 'nt'），使用 cls 命令清屏；否则（Linux/macOS 等），使用 clear 命令清屏。
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """
    打印教程标题
    """
    print("="*60)
    print(f"{title:^60}")
    print("="*60)
    print("\n欢迎使用Pandas数据分析教程！本教程将帮助您学习Pandas库的基本用法和高级功能。")
    print("通过这些示例，您将了解如何使用Pandas进行数据操作、清洗、分析和可视化。")

def print_menu():
    """
    打印菜单选项
    """
    print("\n请选择要运行的教程模块：")
    print("1. Pandas基础知识 (Series, DataFrame, 索引和选择)")
    print("2. Pandas数据操作和转换 (合并, 重塑, 分组, 时间序列)")
    print("3. Pandas数据清洗和预处理 (缺失值, 重复值, 异常值, 数据转换)")
    print("4. Pandas数据可视化 (基本绘图, 统计图表, 多子图)")
    print("0. 退出教程")

def run_module(module_name):
    """
    运行指定的模块

    参数:
        module_name: 要运行的模块名称
    """
    try:
        # 导入模块
        module = importlib.import_module(module_name)

        # 清屏并运行模块的main函数
        clear_screen()
        module.main()

        input("\n按Enter键返回主菜单...")
    except ImportError:
        print(f"\n错误：无法导入模块 '{module_name}'")
        input("按Enter键继续...")
    except AttributeError:
        print(f"\n错误：模块 '{module_name}' 没有main函数")
        input("按Enter键继续...")
    except Exception as e:
        print(f"\n运行模块时出错：{str(e)}")
        input("按Enter键继续...")

def main():
    """
    主函数，显示菜单并处理用户选择
    """
    while True:
        clear_screen()
        print_header("Pandas 数据分析教程")
        print_menu()

        choice = input("\n请输入选项 (0-4): ")

        if choice == '0':
            clear_screen()
            print_header("感谢使用Pandas教程！")
            print("\n希望这些教程对您学习Pandas有所帮助！")
            print("再见！")
            break
        elif choice == '1':
            run_module("pandas_basics")
        elif choice == '2':
            run_module("pandas_data_manipulation")
        elif choice == '3':
            run_module("pandas_data_cleaning")
        elif choice == '4':
            run_module("pandas_visualization")
        else:
            print("\n无效的选项，请重新选择")
            input("按Enter键继续...")

if __name__ == "__main__":
    # 确保当前目录在Python路径中，以便能够导入模块
    # 获取当前脚本所在的绝对目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 检查当前目录是否已在Python模块搜索路径(sys.path)中
    if current_dir not in sys.path:
        # 如果不在，则将当前目录添加到Python模块搜索路径中
        sys.path.append(current_dir)

    main()
