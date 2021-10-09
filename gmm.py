import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm


class GMM():
    def __init__(self, k, mu, sigma, alpha):
        self.k = k
        self.gauss = []
        for i in range(k):
            self.gauss.append({"mu": mu[i], "sigma": sigma[i]})
        self.alpha = alpha

    def _f_gauss(self, x, mu, sigma):
        return 1. / sqrt(2. * pi) / sigma * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def _F_gauss(self, x, mu, sigma):
        xx = (x - mu) / sigma
        return norm.cdf(xx)

    def f(self, x):
        ret = 0.
        for i in range(self.k):
            ret += self._f_gauss(x, self.gauss[i]["mu"], self.gauss[i]["sigma"]) * self.alpha[i]
        return ret

    def cdf(self, x):
        ret = 0.
        for i in range(self.k):
            ret += self._F_gauss(x, self.gauss[i]["mu"], self.gauss[i]["sigma"]) * self.alpha[i]
        return ret

    def ppf(self, x):
        assert x >= 0. and x <= 1.
        l = -10000000
        r = 10000000
        while r - l > 1e-7:
            mid = (l + r) / 2
            if self.cdf(mid) <= x:
                l = mid
            else:
                r = mid
        return (l + r) / 2

    def plot(self):
        x = np.linspace(-10, 10, 100000)
        p = self.f(x)
        plt.plot(x, p)
        plt.show()


if __name__ == '__main__':
    gmm = GMM(2, [0, 3], [3, 0.8], [0.8, 0.2])
    print(gmm.f(2))
    gmm.plot()
