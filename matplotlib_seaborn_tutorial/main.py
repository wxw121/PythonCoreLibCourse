"""
Matplotlib & Seaborn教程主程序
本程序作为教程的入口点，提供交互式菜单来运行不同的示例。
"""

import os
import sys
from typing import Callable, Dict, List

def clear_screen():
    """清理控制台屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """打印程序头部信息"""
    print("="*50)
    print("Matplotlib & Seaborn 教程示例程序")
    print("="*50)
    print("\n")

def print_menu(options: Dict[str, str]):
    """打印菜单选项"""
    print("\n可用的示例:")
    print("-" * 30)
    for key, value in options.items():
        print(f"{key}. {value}")
    print("-" * 30)
    print("q. 退出程序")
    print("\n")

def get_user_choice(options: Dict[str, str]) -> str:
    """获取用户输入的选择"""
    while True:
        choice = input("请选择要运行的示例 (输入选项编号): ").strip().lower()
        if choice == 'q':
            return choice
        if choice in options:
            return choice
        print("无效的选择，请重试。")

def run_example(module_name: str, function_name: str = "run_example"):
    """
    动态导入并运行示例

    Args:
        module_name: 模块名称
        function_name: 要运行的函数名称，默认为'run_example'
    """
    try:
        # 动态导入模块
        module = __import__(module_name)
        # 获取并运行示例函数
        example_function = getattr(module, function_name)
        example_function()
    except ImportError:
        print(f"错误：无法导入模块 {module_name}")
    except AttributeError:
        print(f"错误：模块 {module_name} 中没有找到函数 {function_name}")
    except Exception as e:
        print(f"运行示例时出错：{str(e)}")

    input("\n按回车键继续...")

def main():
    """主程序入口"""
    # 示例选项
    examples = {
        "1": "Matplotlib基础绘图 (matplotlib_basics)",
        "2": "Matplotlib高级特性 (matplotlib_advanced)",
        "3": "Seaborn基础绘图 (seaborn_basics)",
        "4": "Seaborn高级特性 (seaborn_advanced)",
        "5": "自定义样式和主题 (custom_styles)"
    }

    # 模块名映射
    module_map = {
        "1": "matplotlib_basics",
        "2": "matplotlib_advanced",
        "3": "seaborn_basics",
        "4": "seaborn_advanced",
        "5": "custom_styles"
    }

    while True:
        clear_screen()
        print_header()
        print_menu(examples)

        choice = get_user_choice(examples)
        if choice == 'q':
            print("\n感谢使用！再见！")
            break

        clear_screen()
        print(f"\n正在运行: {examples[choice]}")
        print("=" * 50)
        run_example(module_map[choice])

if __name__ == "__main__":
    # 确保示例模块可以被导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
