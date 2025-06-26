"""
Matplotlib配置文件
在任何其他matplotlib相关操作之前导入此文件
"""
import matplotlib
# 设置非交互式后端
matplotlib.use('Agg')
# 禁用所有交互式功能
matplotlib.interactive(False)
# 确保不使用任何tkinter相关功能
import os
os.environ['DISPLAY'] = ''  # 禁用X11 display
