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
        if param[0] == 'incr_size':
            incr_size = int(param[1])
        if param[0] == 'relabel_size':
            relabel_size = int(param[1])
        if param[0] == 'query_times':
            query_times = int(param[1])
        if param[0] == 'num_clss':
            num_clss = int(param[1])
        if line == 'Query 0\n':
            break
    query_batch_size = incr_size + relabel_size
    content = f.readlines()


a_performance = []
am_performance = []
r_performance = []

active = True
xs_a = [init_size]


for i, line in enumerate(content):

    if line.startswith('Random Query'):
        per = float(content[i+retrain_epochs+1][-8:-3])
        r_performance.append(per)

    if line.startswith('Majority'):
        per = float(line[-8:-3])
        if active:
            am_performance.append(per)

    if line.startswith('Classifier 0'):
        per = float(content[i+retrain_epochs+1][-8:-3])
        a_performance.append(per)

    if line.startswith('[torch'):
        xs_a.append(xs_a[-1]+int(line[27:-4])+relabel_size)

print(a_performance)
print(r_performance)

xs = np.arange(
    init_size, init_size+query_batch_size*query_times+1, query_batch_size)
xs_a = xs_a[:-1]

if incr_size == 0:
    xs_a = xs

plt.plot(xs_a, a_performance, label='active')
plt.plot(xs_a, am_performance, label='active majority')
plt.plot(xs, r_performance, label='random')
plt.xlabel('number of queryied samples')
plt.ylabel('performace (%)')
plt.legend()

if len(sys.argv) >= 3:
    plt.savefig(sys.argv[2])

plt.show()
