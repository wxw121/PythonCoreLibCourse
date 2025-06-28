#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch教程主入口

本模块作为PyTorch教程的主入口，允许用户选择运行不同的教程模块。
包含的教程模块有：
1. PyTorch基础
2. PyTorch神经网络
3. PyTorch计算机视觉
4. PyTorch自然语言处理
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

# 导入各个教程模块
try:
    from pytorch_basics import main as basics_main
    from pytorch_neural_networks import main as nn_main
    from pytorch_computer_vision import main as cv_main
    from pytorch_nlp import main as nlp_main
except ImportError:
    # 如果直接运行此文件，添加当前目录到路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from pytorch_basics import main as basics_main
    from pytorch_neural_networks import main as nn_main
    from pytorch_computer_vision import main as cv_main
    from pytorch_nlp import main as nlp_main


def print_header() -> None:
    """打印教程标题"""
    print("\n" + "=" * 80)
    print("PyTorch核心教程".center(80))
    print("=" * 80)


def print_menu() -> None:
    """打印菜单选项"""
    print("\n请选择要运行的教程模块：")
    print("1. PyTorch基础 (张量、自动求导、GPU加速、数据加载)")
    print("2. PyTorch神经网络 (线性层、激活函数、损失函数、优化器)")
    print("3. PyTorch计算机视觉 (CNN、图像分类、数据增强、迁移学习)")
    print("4. PyTorch自然语言处理 (文本处理、词嵌入、RNN、Transformer)")
    print("5. 运行所有教程")
    print("0. 退出")


def run_tutorial(choice: int) -> None:
    """
    运行选择的教程模块

    Args:
        choice: 用户的选择
    """
    if choice == 1:
        print("\n正在运行 PyTorch基础 教程...")
        basics_main()
    elif choice == 2:
        print("\n正在运行 PyTorch神经网络 教程...")
        nn_main()
    elif choice == 3:
        print("\n正在运行 PyTorch计算机视觉 教程...")
        cv_main()
    elif choice == 4:
        print("\n正在运行 PyTorch自然语言处理 教程...")
        nlp_main()
    elif choice == 5:
        print("\n正在运行所有教程...")
        basics_main()
        nn_main()
        cv_main()
        nlp_main()
    else:
        print("\n无效的选择，请重新选择。")


def interactive_mode() -> None:
    """交互式模式，显示菜单并处理用户输入"""
    print_header()

    while True:
        print_menu()
        try:
            choice = int(input("\n请输入选项编号: "))
            if choice == 0:
                print("\n感谢使用PyTorch教程！再见！")
                break
            run_tutorial(choice)
            input("\n按Enter键继续...")
        except ValueError:
            print("\n请输入有效的数字选项。")
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。感谢使用PyTorch教程！")
            break


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PyTorch教程")
    parser.add_argument(
        "module",
        nargs="?",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="要运行的教程模块 (1=基础, 2=神经网络, 3=计算机视觉, 4=NLP, 5=全部)"
    )
    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    if args.module:
        # 命令行模式
        print_header()
        run_tutorial(args.module)
    else:
        # 交互式模式
        interactive_mode()


if __name__ == "__main__":
    main()
