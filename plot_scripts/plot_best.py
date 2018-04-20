import sys
import numpy as np
import matplotlib.pyplot as plt


with open(sys.argv[1]) as f:
    for line in f:
        param = line.split(':')
        if param[0] == 'retrain_epochs':
            retrain_epochs = int(param[1])
        if param[0] == 'init_size':
            init_size = int(param[1])
        if param[0] == 'query_batch_size':
            query_batch_size = int(param[1])
        if param[0] == 'incr_times':
            incr_times = int(param[1])
        if param[0] == 'num_clss':
            num_clss = int(param[1])
        if line == 'incr 0\n':
            break
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
           if line != 'Use logistic exceptionally\n']


for i, line in enumerate(content):

    if line.startswith('Active Query'):
        active = True
        a_performance.append(0)
        aa_performance.append(0)

    if line.startswith('Random Query'):
        active = False
        r_performance.append(0)
        ra_performance.append(0)

    if line.startswith('Majority'):
        per = float(line[-8:-3])
        if active:
            am_performance.append(per)
        else:
            rm_performance.append(per)

    if line.startswith('classifier'):
        per = float(content[i+retrain_epochs+1][-8:-3])
        if active:
            a_performance[-1] = max(a_performance[-1], per)
            aa_performance[-1] += per
        else:
            r_performance[-1] = max(r_performance[-1], per)
            ra_performance[-1] += per

    if line.startswith('[torch'):
        xs_a.append(xs_a[-1]+int(line[27:-4]))

aa_performance = [per/num_clss for per in aa_performance]
ra_performance = [per/num_clss for per in ra_performance]

print(r_performance)
print(a_performance)

xs = np.arange(
    init_size, init_size+query_batch_size*incr_times+1, query_batch_size)

xs_a = xs_a[:-1]
if len(xs_a) == 0:
    xs_a = xs

plt.plot(xs, r_performance, label='random best')
plt.plot(xs, ra_performance, label='random average')
if rm_performance != []:
    plt.plot(xs, rm_performance, label='random majority')
    print(rm_performance)
plt.plot(xs_a, a_performance, label='active best')
plt.plot(xs_a, aa_performance, label='active average')
if am_performance != []:
    plt.plot(xs_a, am_performance, label='active majority')
    print(am_performance)
plt.xlabel('number of queryied samples')
plt.ylabel('performace (%)')
plt.legend()

if len(sys.argv) >= 3:
    plt.savefig(sys.argv[2])

plt.show()
