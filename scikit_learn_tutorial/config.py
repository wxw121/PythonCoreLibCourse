# 导入matplotlib配置（必须在任何其他matplotlib导入之前）
from matplotlib_config import *
import matplotlib
import matplotlib.pyplot as plt

def set_matplotlib_chinese():
    """
    设置Matplotlib以正确显示中文字符和负号
    在需要显示中文的模块中调用此函数
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    print("中文显示设置已启用")
