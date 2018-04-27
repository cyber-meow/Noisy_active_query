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

active = False
random = False
first = True

for i, line in enumerate(content):

    if line.startswith('Active Query'):
        active = True

    if line.startswith('Random Query'):
        random = True

    if line.startswith('Test set'):
        per = float(line[-8:-3])
        if active:
            if first:
                first = False
            else:
                a_performance.append(per)
                first = True
                active = False
        if random:
            if first:
                first = False
            else:
                r_performance.append(per)
                first = True
                random = False

print(r_performance)
print(a_performance)

xs = np.arange(
    init_size, init_size+query_batch_size*incr_times+1, query_batch_size)


if r_performance != []:
    plt.plot(xs, r_performance, label='random')
plt.plot(xs, a_performance, label='active')

plt.xlabel('number of queried samples')
plt.ylabel('performace (%)')
plt.legend()

if len(sys.argv) >= 3:
    plt.savefig(sys.argv[2])

plt.show()
