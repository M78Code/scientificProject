import matplotlib.pyplot as plt
import random

from pylab import mpl


# 绘制基本图像
def _fun_1():
    # 设置显示中文字体
    mpl.rcParams["font.sans-serif"] = ['SimHei']
    # 设置正常显示符号
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制基本图像
    # 0.准备数据
    x = range(60)
    y_shanghai = [random.uniform(15, 18) for i in x]
    # 1.创建画布
    plt.figure(figsize=(20, 8), dpi=100)
    # 2.绘制图像
    plt.plot(x, y_shanghai)
    # 3.图像显示
    plt.show()


# 实现一些其他功能
def _fun_2():
    # 实现一些其他功能
    # 0.准备数据
    x = range(60)
    y_shanghai = [random.uniform(15, 18) for i in x]
    # 1.创建画布
    plt.figure(figsize=(20, 8), dpi=100)
    # 2.绘制图像
    plt.plot(x, y_shanghai)

    # 2.1 添加x, y轴刻度
    # 设置x, y 轴刻度
    x_ticks_label = ['11点{}'.format(i) for i in x]
    y_ticks = range(40)

    # 修改x, y轴坐标刻度显示
    plt.xticks(x[::5], x_ticks_label[::5])
    plt.yticks(y_ticks[::5])

    # 2.2 添加网格显示
    plt.grid(True, linestyle="--", alpha=1)

    # 2.3 添加描述信息
    plt.xlabel("时间")
    plt.ylabel("温度")
    plt.title("中午11点-12点某城市温度变化图", fontsize=20)

    # 2.4 图像保存
    plt.savefig("./test.png")

    # 3.图像显示
    plt.show()


# _fun_2()

# 在一个坐标系中绘制多个图像
def _fun_3():
    # 0.准备数据
    x = range(60)
    y_shanghai = [random.uniform(15, 18) for i in x]
    y_beijing = [random.uniform(1, 3) for i in x]

    # 1.创建画布
    plt.figure(figsize=(20, 8), dpi=100)

    # 2.绘制图像
    plt.plot(x, y_shanghai, label="上海")
    plt.plot(x, y_beijing, color="r", linestyle="--", label="北京")

    # 2.1 添加x,y轴刻度
    # 设置x,y轴刻度
    x_ticks_label = ["11点{}分".format(i) for i in x]
    y_ticks = range(40)

    # 修改x,y轴坐标刻度显示
    # plt.xticks(x_ticks_label[::5]) # 坐标刻度不可以直接通过字符串进行修改
    plt.xticks(x[::5], x_ticks_label[::5])
    plt.yticks(y_ticks[::5])

    # 2.2 添加网格显示
    plt.grid(True, linestyle="--", alpha=1)

    # 2.3 添加描述信息
    plt.xlabel("时间")
    plt.ylabel("温度")
    plt.title("中午11点-12点某城市温度变化图", fontsize=20)

    # 2.4 图像保存
    plt.savefig("./test.png")

    # 2.5 显示图例
    plt.legend(loc=0)

    # 3.图像显示
    plt.show()


_fun_3()
