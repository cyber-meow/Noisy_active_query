import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

import dataset
from classifier import Classifier


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


class ToyClassifier(Classifier):

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
