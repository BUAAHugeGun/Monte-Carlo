import matplotlib.pyplot as plt
import math
from matplotlib.font_manager import FontProperties
import numpy as np

if __name__ == "__main__":
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'color': 'black',
            'size': 14
            }
    t1_mse = [0.0028345, 2.72E-06, 2.97E-08, 3.6352E-09, 3.08E-11, 4.09E-12]
    t2_mse = [7.24E-04, 1.57E-06, 1.41E-08, 2.28E-09, 1.72E-11, 2.12E-12]
    t1_entropy = [0.3086, 0.02949, 0.0071947, 0.002007, 0.000393229, 0.00023946]
    t2_entropy = [0.10567, 0.01808, 0.00309, 0.001719, 1.94E-04, 7.54E-05]

    t1_mse = np.array(t1_mse)
    t2_mse = np.array(t2_mse)
    t1_entropy = np.array(t1_entropy)
    t2_entropy = np.array(t2_entropy)

    t1_mse = -np.log2(t1_mse)
    t2_mse = -np.log2(t2_mse)
    t1_entropy = -np.log2(t1_entropy)
    t2_entropy = -np.log2(t2_entropy)

    # for l in msssim:
    #     for i in range(len(l)):
    #         l[i]=-10*math.log10(1-l[i])
    n = [100, 1000, 5000, 10000, 50000, 100000]

    # plt.axhline(y=37.3588, color='red')
    # plt.axhline(y=10.8474, color='red')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.figure(figsize=(10, 4.5))
    plt.grid()
    plt.plot(n, t1_entropy, color="dodgerblue", marker="o", label="t1")
    plt.plot(n, t2_entropy, color="orange", marker="v", label="t2")
    plt.ylabel("-log(KL)", fontdict=font)
    plt.xlabel("times", fontdict=font)

    plt.show()
