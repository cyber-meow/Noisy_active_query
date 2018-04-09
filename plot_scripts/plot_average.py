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
r_performance = []
active = True


content = [line for line in content
           if line != 'use logistic exceptionally\n']


for i, line in enumerate(content):

    if line.startswith('epoch'):
        epoch = int(line[6:-1])

    if line.startswith('Active Query'):
        active = True
        a_performance.append(0)
        n_clss = 0

    if line.startswith('Random Query'):
        active = False
        r_performance.append(0)
        n_clss = 0

    if line.startswith('classifier'):
        per = float(content[i+epoch][-8:-3])
        if active:
            a_performance[-1] += per
            n_clss += 1
        else:
            r_performance[-1] += per
            n_clss += 1


a_performance = [per/n_clss for per in a_performance]
r_performance = [per/n_clss for per in r_performance]

print(a_performance)
print(r_performance)

xs = np.arange(
    init_size, init_size+query_batch_size*incr_times+1, query_batch_size)

plt.plot(xs, a_performance, label='active')
plt.plot(xs, r_performance, label='random')
plt.xlabel('number of queryied samples')
plt.ylabel('performace (%)')
plt.legend()

if len(sys.argv) >= 3:
    plt.savefig(sys.argv[2])

plt.show()
