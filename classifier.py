import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import settings


class Classifier(object):

    def __init__(self, model, pho_p=0, pho_n=0, lr=5e-3):
        self.model = model
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-2)
        self.pho_p = pho_p
        self.pho_n = pho_n
        self.counter = 0

    def train(self, labeled_set, test_set,
              batch_size, retrain_epochs,
              convex_epochs=None, used_size=None, test_on_train=False):
        self.model.train()
        if used_size is None:
            train_loader = data.DataLoader(
                labeled_set, batch_size=batch_size,
                shuffle=True, num_workers=2)
        else:
            indices = np.random.choice(
                len(labeled_set), used_size, replace=False)
            train_loader = data.DataLoader(
                labeled_set, batch_size=batch_size,
                sampler=data.sampler.SubsetRandomSampler(indices),
                num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=True, num_workers=2)
        for epoch in range(retrain_epochs):
            if convex_epochs is not None and epoch >= convex_epochs:
                self.train_step(train_loader, epoch, convex_loss=False)
            else:
                self.train_step(train_loader, epoch)
            if test_on_train:
                self.test(train_loader, 'Train')
            self.test(test_loader, 'Test')

    def train_step(self, train_loader, epoch, convex_loss=True):
        total_loss = 0
        for batch_idx, (x, target, w) in enumerate(train_loader):
            self.optimizer.zero_grad()
            x, target, w = (
                Variable(x).type(settings.dtype),
                Variable(target).type(settings.dtype),
                Variable(w).type(settings.dtype))
            output = self.model(x)
            _, loss = self.compute_loss(output, target, convex_loss, w)
            total_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()
        if (batch_idx+1) % 100 == 0:
            print(
                '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx+1, total_loss))

    def basic_loss(self, fx, convex_loss=True):
        if convex_loss:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)

    def compute_loss(self, output, target, convex_loss=True, w=None):
        a = (self.pho_p - self.pho_n)/2
        b = (self.pho_p + self.pho_n)/2
        pho_y = a * target + b
        pho_ny = self.pho_p + self.pho_n - pho_y
        loss = self.basic_loss(target*output, convex_loss)
        loss = (1-pho_ny+pho_y) * loss
        if w is not None:
            assert w.shape == loss.shape
            loss *= w
        total_loss = torch.sum(loss)
        return loss, total_loss

    def test(self, test_loader, set_name):
        self.model.eval()
        test_loss = 0
        correct = 0
        for item in test_loader:
            x, target = item[0], item[1]
            x, target = (
                Variable(x).type(settings.dtype),
                Variable(target).type(settings.dtype))
            output = self.model(x)
            pred = torch.sign(output)
            correct += torch.sum(pred.eq(target).float()).data[0]
        test_loss /= len(test_loader.dataset)
        print(
            '{} set: Accuracy: {}/{} ({:.2f}%)'.format(
                set_name, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
