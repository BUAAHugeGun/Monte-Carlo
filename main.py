import matplotlib.pyplot as plt
import numpy as np
from model import *
from random import random
from multiprocessing import Pool
from tqdm import tqdm
import math
import os
from criterion import *


class T():
    def __init__(self, model: base_model, n, avg=True):
        self.avg = avg
        self.n = n
        self.model = model

    def readx(self):
        f = open(str(self.n) + ".txt", mode="r")
        ll = f.readline().split(" ")
        self.n = int(ll[0])
        self.blk = int(ll[1])
        print("reading sampled data: (n={}, blk={})".format(self.n, self.blk))
        ll = f.readline().split(" ")[:-1]
        self.x = [float(p) for p in ll]
        self.x = [self.x[i:i + self.blk] for i in range(0, self.n, self.blk)]

    def sample(self, blk, cover=False):
        self.blk = blk
        ran = []
        x = []
        if os.path.exists(str(self.n) + ".txt") and not cover:
            self.readx()
            return
        for i in range(self.n):
            ran.append(random())
        p = Pool(11)
        x = p.map(self.model.ppf, ran)

        f = open(str(self.n) + ".txt", mode="w")
        f.write(str(self.n) + " ")
        f.write(str(self.blk) + "\n")
        for y in x:
            f.write(str(y) + " ")
        self.x = [x[i:i + blk] for i in range(0, self.n, blk)]
        # plt.hist(x, 100, density=True)
        # self.model.plot()
        # plt.show()

    def get_T1(self):
        x = [sum(y) for y in self.x]
        self.t1 = sorted(x)
        plt.hist(x, 100, density=True)

    def t1f(self, x):
        X = self.t1

        def erfen(a):
            if a > X[-1]:
                return len(X) + 1
            l = 0
            r = len(X) - 1
            while l < r:
                mid = (l + r) // 2
                if X[mid] <= a:
                    l = mid + 1
                else:
                    r = mid
            return l

        return (erfen(x + 0.5) - erfen(x - 0.5)) / len(X)

    def t2f(self, x):
        X = self.t2

        def erfen(a):
            if a > X[-1]:
                return len(X) + 1
            l = 0
            r = len(X) - 1
            while l < r:
                mid = (l + r) // 2
                if X[mid] <= a:
                    l = mid + 1
                else:
                    r = mid
            return l

        return (erfen(x + 0.5) - erfen(x - 0.5)) / len(X)

    def get_T2(self):
        x = [max(y) for y in self.x]
        self.t2 = sorted(x)
        plt.hist(x, 100, density=True)

    def plot(self):
        #self.model.plot()
        plt.show()


def FT_12(x):
    return (np.exp(-4 - x ** 2 / 16) * (np.exp(4) + np.cosh(x))) / (8 * np.sqrt(math.pi))


def log_FT_12(x):
    p = (np.exp(-4 - x ** 2 / 16) * (np.exp(4) + np.cosh(x))) / (8 * np.sqrt(math.pi))
    return p * np.log2(p)


def FT_22(x):
    return np.exp(-1. / 8 * (4 + x) ** 2) * (1 + np.exp(2 * x)) * \
           (2 + erf((-4 + x) / (2 * sqrt(2))) + erf((4 + x) / (2 * sqrt(2)))) / (8 * sqrt(2 * pi))


def log_FT_22(x):
    p = np.exp(-1. / 8 * (4 + x) ** 2) * (1 + np.exp(2 * x)) * \
        (2 + erf((-4 + x) / (2 * sqrt(2))) + erf((4 + x) / (2 * sqrt(2)))) / (8 * sqrt(2 * pi))
    return p * log2(p)


if __name__ == "__main__":
    t = T(GMM(2, [-4, 4], [2, 2], [0.5, 0.5]), 20000)
    t.sample(2)
    # t.get_T1()
    t.get_T2()

    x = np.linspace(-10, 10, 100000)
    p = []
    for xx in x:
        p.append(FT_22(xx))
    plt.plot(x, p)
    t.plot()
    # print(MSE(FT_12, t.t1))
    # print(entropy(FT_12, log_FT_12, t.t1, t.t1f))
    # print("")
    # print(MSE(FT_22, t.t2))
    # print(entropy(FT_22, log_FT_22, t.t2, t.t2f))
