import matplotlib.pyplot as plt
import numpy as np
from model import *
from random import random
from multiprocessing import Pool
from tqdm import tqdm


class T1():
    def __init__(self, model: base_model, n, avg=True):
        self.avg = avg
        self.n = n
        self.model = model

    def sample(self, blk):
        self.blk = blk
        ran = []
        x = []
        for i in range(self.n):
            ran.append(random())
        p = Pool(8)
        x = p.map(self.model.ppf, ran)
        self.x = [x[i:i + blk] for i in range(0, self.n, blk)]
        print(len(self.x))
        # plt.hist(x, 100, density=True)
        # self.model.plot()
        # plt.show()

    def get_T1(self):
        x = [sum(y) / self.blk for y in self.x]
        plt.hist(x, 100, density=True)
        plt.show()


if __name__ == "__main__":
    t = T1(GMM(2, [0, 3], [3, 0.8], [0.8, 0.2]), 1000000)
    t.sample(100)
    t.get_T1()
