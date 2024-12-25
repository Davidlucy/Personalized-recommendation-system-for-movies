import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, data):
        self.data = data
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号