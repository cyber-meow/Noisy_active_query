import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn import datasets

import dataset
from active_query import IWALQuery
from classifier import Classifier, majority_vote


moons = True
n_positive = 5000
n_negative = 5000
n = n_positive + n_negative

pho_p = 0.5
pho_n = 0
pho_p_c = pho_p
pho_n_c = pho_n

learning_rate = 5e-3
weight_decay = 1e-3

convex_epochs = 4000
retrain_epochs = 16000
final_epochs = 24000

num_clss = 2
init_size = 90
kcenter = True

used_size = 80
incr_times = 8
query_batch_size = 6
reduced_sample_size = 1

init_weight = 1
weight_ratio = 2


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def border_func(self, x, y):
        inp = Variable(torch.from_numpy(np.array([x, y])).float())
        return self.forward(inp).data.numpy()

    def plot_boundary(self, ax, **kwargs):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xs = np.linspace(xmin, xmax, 100)
        ys = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(xs, ys)
        border_func = np.vectorize(self.border_func)
        cont = plt.contour(xv, yv, border_func(xv, yv), [0], **kwargs)
        return cont


conts_dy = []


class ToyClassifier(Classifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

    def train(self, labeled_set, test_set, retrain_epochs,
              convex_epochs=None, used_size=None,
              test_interval=1, test_on_train=False):

        self.model.train()
        self.init_optimizer()

        if used_size is None:
            train_set = labeled_set
        else:
            indices = torch.from_numpy(np.random.choice(
                len(labeled_set), used_size, replace=False))
            train_set = dataset.WeightedTensorDataset(
                labeled_set.data_tensor[indices],
                labeled_set.target_tensor[indices],
                labeled_set.weight_tensor[indices])

        for epoch in range(retrain_epochs):

            if convex_epochs is not None and epoch >= convex_epochs:
                if self.last_accuracy < 75:
                    # print('Use logistic exceptionally')
                    self.train_step(train_set, epoch)
                else:
                    self.train_step(train_set, epoch, convex_loss=False)
            else:
                self.train_step(train_set, epoch)

            if (epoch+1) % test_interval == 0 or epoch+1 == retrain_epochs:
                sys.stdout.write('Epoch: {}  '.format(epoch))
                if test_on_train:
                    self.test(train_set, 'Train')
                self.test(test_set, 'Test')

    def train_step(self, train_set, epoch, convex_loss=True):
        # total_loss = 0
        self.optimizer.zero_grad()
        x = Variable(train_set.data_tensor).float()
        target = Variable(train_set.target_tensor).float()
        w = Variable(train_set.weight_tensor).float()
        output = self.model(x)
        _, loss = self.compute_loss(output, target, convex_loss, w)
        loss.backward()
        self.optimizer.step()

    def test(self, test_set, set_name):
        self.model.eval()
        x = Variable(test_set.data_tensor).float()
        target = Variable(test_set.target_tensor).float()
        output = self.model(x)
        pred = torch.sign(output)
        correct = torch.sum(pred.eq(target).float()).data[0]

        # if (100 * correct / len(test_set) < self.last_accuracy
        #         and set_name == 'Train'):
        #     conts_dy.append(self.model.plot_boundary(ax))
        #     if len(conts_dy) > 2:
        #         for coll in conts_dy[0].collections:
        #             coll.remove()
        #         del conts_dy[0]
        #     plt.pause(0.05)

        self.last_accuracy = 100 * correct / len(test_set)
        self.best_accuracy = max(self.best_accuracy, self.last_accuracy)
        self.use_logistic_threshold = max(
            self.use_logistic_threshold, self.best_accuracy-5)
        print(
            '{} set: Accuracy: {}/{} ({:.2f}%)'.format(
                set_name, correct, len(test_set),
                self.last_accuracy))


def create_new_classifier():
    model = Net()
    cls = ToyClassifier(
            model,
            pho_p=pho_p_c,
            pho_n=pho_n_c,
            lr=learning_rate)
    return cls


if os.path.exists('datasets/toy/train_data.npy') and False:
    x_all = np.load('datasets/toy/train_data.npy')
    y_all_corrupted = np.load('datasets/toy/train_labels.npy')

else:
    if moons:
        x_all, y_all = datasets.make_moons(n, noise=0.07)
    else:
        x_all, y_all = datasets.make_circles(n, noise=0.03)
    y_all = (y_all*2-1).reshape(-1, 1)

    y_all_corrupted = dataset.label_corruption(y_all, pho_p, pho_n)

    np.save('datasets/toy/train_data', x_all)
    np.save('datasets/toy/train_labels', y_all_corrupted)

if kcenter:
    unlabeled_set, labeled_set = dataset.datasets_initialization_kcenter(
        x_all, y_all_corrupted, init_size, init_weight)
else:
    unlabeled_set, labeled_set = dataset.datasets_initialization(
        x_all, y_all_corrupted, init_size, init_weight)


if os.path.exists('datasets/toy/test_data.npy') and False:
    x_test = np.load('datasets/toy/test_data.npy')
    y_test = np.load('datasets/toy/test_labels.npy')
else:
    if moons:
        x_test, y_test = datasets.make_moons(n, noise=0.07)
    else:
        x_test, y_test = datasets.make_circles(n, noise=0.03)
    y_test = (y_test*2-1).reshape(-1, 1)
    np.save('datasets/toy/test_data', x_test)
    np.save('datasets/toy/test_labels', y_test)

test_set = torch.utils.data.TensorDataset(
    torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())


fig, ax = plt.subplots()

plt.ion()
plt.show()

negative_samples = x_all[y_all.reshape(-1) == 1]
positive_samples = x_all[y_all.reshape(-1) == -1]

px, py = np.array(positive_samples).T
nx, ny = np.array(negative_samples).T
plt.scatter(px, py, color='mistyrose', s=3)
plt.scatter(nx, ny, color='turquoise', s=3)
plt.pause(0.05)

x_init = labeled_set.data_tensor.numpy()
y_init = labeled_set.target_tensor.numpy().reshape(-1)

cx, cy = np.array(x_init).T
plt.scatter(cx, cy, s=3, color='yellow')
cx, cy = x_init[y_init == -1].T
plt.scatter(cx, cy, s=3, c='black', alpha=0.2)
plt.pause(0.05)


conts = []
cm = plt.get_cmap('gist_rainbow')

clss = [create_new_classifier() for _ in range(num_clss)]
IWALQuery = IWALQuery()


for incr in range(incr_times+1):

    print('\nincr {}'.format(incr))

    for i, cls in enumerate(clss):
        print('classifier {}'.format(i))
        cls.train(labeled_set, test_set, retrain_epochs,
                  convex_epochs, used_size,
                  test_interval=3000, test_on_train=True)
        if incr >= 1:
            for coll in conts[0].collections:
                coll.remove()
            del conts[0]
        conts.append(cls.model.plot_boundary(ax, colors=[cm(i/num_clss)]))
        plt.pause(0.05)
    if num_clss > 1:
        majority_vote(clss, test_set)

    if incr < incr_times:
        x_selected, y_selected, _ = IWALQuery.query(
            unlabeled_set, labeled_set, query_batch_size, clss, weight_ratio)
        used_size += len(x_selected) - reduced_sample_size

        x_selected = x_selected.numpy()
        y_selected = y_selected.numpy().reshape(-1)
        sx, sy = x_selected.T
        plt.scatter(sx, sy, s=10, label='{}'.format(incr))
        sx, sy = x_selected[y_selected == -1].T
        plt.scatter(sx, sy, s=25, c='black', alpha=0.2)
        plt.legend()
        plt.pause(0.05)


if incr_times > 0:
    cls = create_new_classifier()
    cls.train(labeled_set, test_set, final_epochs, convex_epochs,
              test_interval=3000)
    cls.model.plot_boundary(ax, colors=['black'])
    plt.pause(0.05)


while not plt.waitforbuttonpress(1):
    pass
