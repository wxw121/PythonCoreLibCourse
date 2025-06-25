"""
Matplotlib & Seaborn 全局配置模块
此模块包含所有全局配置设置，如字体、样式等
"""

import matplotlib.pyplot as plt

def setup_chinese_display():
    """
    设置Matplotlib以正确显示中文字符和负号
    在需要显示中文的模块中调用此函数
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    print("中文显示设置已启用")

def setup_global_style(style=None):
    """
    设置全局绘图样式

    Args:
        style: 要使用的样式名称，如果为None则使用默认样式
    """
    if style:
        try:
            plt.style.use(style)
            print(f"已应用样式: {style}")
        except:
            print(f"样式 '{style}' 不可用，使用默认样式")

def reset_to_defaults():
    """
    重置所有设置为Matplotlib默认值
    """
    plt.rcdefaults()
    print("已重置为默认设置")
