import sys
import numpy as np
import matplotlib.pyplot as plt


with open(sys.argv[1]) as f:
    init_size = int(f.readline())
    f.readline()
    query_batch_size = int(f.readline())
    incr_times = int(f.readline())
    content = f.readlines()


a_performance = []
aa_performance = []
am_performance = []
r_performance = []
ra_performance = []
rm_performance = []

active = True
xs_a = [init_size]


content = [line for line in content
           if line != 'use logistic exceptionally\n']


for i, line in enumerate(content):

    if line.startswith('epoch'):
        epoch = int(line[6:-1])

    if line.startswith('Active Query'):
        active = True
        a_performance.append(0)
        aa_performance.append(0)
        n_clss = 0

    if line.startswith('Random Query'):
        active = False
        r_performance.append(0)
        ra_performance.append(0)
        n_clss = 0

    if line.startswith('Majority'):
        per = float(line[-8:-3])
        if active:
            am_performance.append(per)
        else:
            rm_performance.append(per)

    if line.startswith('classifier'):
        per = float(content[i+epoch][-8:-3])
        if active:
            a_performance[-1] = max(a_performance[-1], per)
            aa_performance[-1] += per
            n_clss += 1
        else:
            r_performance[-1] = max(r_performance[-1], per)
            ra_performance[-1] += per
            n_clss += 1

    if line.startswith('[torch'):
        xs_a.append(xs_a[-1]+int(line[27:-4]))

print(a_performance)
print(r_performance)

aa_performance = [per/n_clss for per in aa_performance]
ra_performance = [per/n_clss for per in ra_performance]

xs = np.arange(
    init_size, init_size+query_batch_size*incr_times+1, query_batch_size)
xs_a = xs_a[:-1]

plt.plot(xs_a, a_performance, label='active best')
plt.plot(xs_a, aa_performance, label='active average')
if am_performance != []:
    plt.plot(xs_a, am_performance, label='active majority')
plt.plot(xs, r_performance, label='random best')
plt.plot(xs, ra_performance, label='random average')
if rm_performance != []:
    plt.plot(xs, rm_performance, label='random majority')
plt.xlabel('number of queryied samples')
plt.ylabel('performace (%)')
plt.legend()

if len(sys.argv) >= 3:
    plt.savefig(sys.argv[2])

plt.show()
