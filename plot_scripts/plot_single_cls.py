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
        if line == 'incr 0\n':
            break
    content = f.readlines()


a_performance = []
r_performance = []


content = [line for line in content
           if line != 'Use logistic exceptionally\n']


for i, line in enumerate(content):

    if line.startswith('Active Query'):
        per = float(content[i+retrain_epochs+2][-8:-3])
        a_performance.append(per)

    if line.startswith('Random Query'):
        per = float(content[i+retrain_epochs+2][-8:-3])
        r_performance.append(per)

print(r_performance)
print(a_performance)

xs = np.arange(
    init_size, init_size+query_batch_size*incr_times+1, query_batch_size)


plt.plot(xs, r_performance, label='random')
plt.plot(xs, a_performance, label='active')

plt.xlabel('number of queried samples')
plt.ylabel('performace (%)')
plt.legend()

if len(sys.argv) >= 3:
    plt.savefig(sys.argv[2])

plt.show()
