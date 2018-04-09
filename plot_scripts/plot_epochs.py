import sys
import matplotlib.pyplot as plt


with open(sys.argv[1]) as f:
    content = f.readlines()


blackbox = False
random = False


for i, line in enumerate(content):

    if line == 'r\n':
        random = True
        r = []
        r_n = 0

    if line == 'b\n':
        b = []
        blackbox = True
        b_n = 0

    if random:
        if line.startswith('classifier'):
            if r != []:
                print(r)
                plt.plot(r, label='random {}'.format(r_n))
                r = []
            r_n = line[-2]
        if line.startswith('Test set'):
            r.append(float(line[-8:-3]))
        if line == '\n':
            print(r)
            plt.plot(r, label='random {}'.format(r_n))
            random = False

    if blackbox:
        if line.startswith('classifier'):
            if b != []:
                print(b)
                plt.plot(b, label='blackbox {}'.format(b_n))
                b = []
            b_n = line[-2]
        if line.startswith('Test set'):
            b.append(float(line[-8:-3]))
        if line == '\n':
            print(b)
            plt.plot(b, label='blackbox {}'.format(b_n))
            blackbox = False


plt.xlabel('number of epochs')
plt.ylabel('performace (%)')
plt.legend()

if len(sys.argv) >= 3:
    plt.savefig(sys.argv[2])

plt.show()
