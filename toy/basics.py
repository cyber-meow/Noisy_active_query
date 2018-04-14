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
              test_interval=1, print_interval=1, test_on_train=False):

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

            if epoch == convex_epochs:
                self.smallest_conf = 100

            if convex_epochs is not None and epoch >= convex_epochs:
                if (self.test_accuracies != []
                        and self.test_accuracies[-1] < 75):
                    # print('Use logistic exceptionally')
                    self.train_step(train_set, epoch)
                else:
                    self.train_step(train_set, epoch, convex_loss=False)
            else:
                self.train_step(train_set, epoch)

            if (epoch+1) % test_interval == 0 or epoch+1 == retrain_epochs:
                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {}  '.format(epoch))
                if test_on_train:
                    self.test(labeled_set, 'Train', to_print)
                self.test(test_set, 'Test', to_print)
                self.find_high_loss_samples(labeled_set)

        self.model = self.critic_model
        if test_on_train:
            self.test(train_set, 'Train')
            self.train_accuracies.pop()
        self.test(test_set, 'Test')
        self.test_accuracies.pop()

    def train_step(self, train_set, epoch, convex_loss=True):
        self.model.train()
        self.optimizer.zero_grad()
        x = Variable(train_set.data_tensor).float()
        target = Variable(train_set.target_tensor).float()
        w = Variable(train_set.weight_tensor).float()
        output = self.model(x)
        _, loss = self.compute_loss(output, target, convex_loss, w)
        loss.backward()
        self.optimizer.step()
