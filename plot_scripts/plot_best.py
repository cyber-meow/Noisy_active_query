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
af_performance = []
am_performance = []
r_performance = []
rf_performance = []
rm_performance = []

active = False
random = False


content = [line for line in content
           if line != 'Use logistic exceptionally\n']


for i, line in enumerate(content):

    if line.startswith('Active Query'):
        active = True
        a_performance.append(0)
        af_performance.append(0)

    if line.startswith('Random Query'):
        random = True
        r_performance.append(0)
        rf_performance.append(0)

    if line.startswith('Majority'):
        per = float(line[-8:-3])
        if active:
            am_performance.append(per)
            active = False
        else:
            rm_performance.append(per)
            random = False

    if line.startswith('Test set'):
        per = float(line[-8:-3])
        if active:
            a_performance[-1] = max(a_performance[-1], per)
            if af_performance[-1] == 0:
                af_performance[-1] = per
        if random:
            r_performance[-1] = max(r_performance[-1], per)
            if rf_performance[-1] == 0:
                rf_performance[-1] = per

print(r_performance)
print(a_performance)
print(rf_performance)
print(af_performance)

xs = np.arange(
    init_size, init_size+query_batch_size*incr_times+1, query_batch_size)

plt.plot(xs, r_performance, label='random best')
plt.plot(xs, rf_performance, label='random first')
# if rm_performance != []:
#     plt.plot(xs, rm_performance, label='random majority')
#     print(rm_performance)
plt.plot(xs, a_performance, label='active best')
plt.plot(xs, af_performance, label='active first')
# if am_performance != []:
#     plt.plot(xs, am_performance, label='active majority')
#     print(am_performance)
plt.xlabel('number of queryied samples')
plt.ylabel('performace (%)')
plt.legend()

if len(sys.argv) >= 3:
    plt.savefig(sys.argv[2])

plt.show()
