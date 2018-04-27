import sys
import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt(sys.argv[1], delimiter=',')
ns = [24, 8, 8, 8, 8, 8]
names = ['Random', 'Bagging', 'Uncertainty', 'Compare with perfect',
         'Random drop 1.5', 'Conf 1 1.5']


def plot_with_variance(m, s, name):
    plt.plot(xs, m, label=name)
    plt.fill_between(xs, m-s, m+s, alpha=0.2)


left = 0
xs = np.arange(100, 201, 20)

for i, n in enumerate(ns):
    d = data[left:left+n]
    left += n
    m = np.mean(d, axis=0)
    s = np.std(d, axis=0)
    plot_with_variance(m, s, names[i])

plt.legend()
plt.show()
