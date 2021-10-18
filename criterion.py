import numpy as np
from scipy import integrate
import math


def MSE(f, x):
    l, r = min(x), max(x)
    x = sorted(x)
    n = len(x) // 10
    data = [[]]
    temp = 0
    blk = (r - l) / n
    for xx in x:
        while xx >= l + temp * blk + blk:
            data.append([])
            temp += 1
        data[temp].append(xx)
    while len(data) < n:
        data.append([])
    ret = 0.
    for i in range(n):
        ret += (integrate.quad(f, l + i * blk, l + (i + 1) * blk)[0] - len(data[i]) / len(x)) ** 2
    return ret / (n // blk)


def entropy(f, log_f, x, tf):
    y=[]
    for xx in x:
        y.append(f(xx))
    y = np.array(y)
    #en = -integrate.quad(log_f, -30, 30)[0]
    #print(en)
    p = []
    for xx in x:
        p.append(tf(xx))
    return -sum(np.log2(y)) / len(x) + sum(np.log2(p)) / len(x)
