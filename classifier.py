import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor


class Classifier(object):

    def __init__(self, model, pho_p=0, pho_n=0):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=5e-3)
        self.pho_p = pho_p
        self.pho_n = pho_n

    def train(self, labeled_set, test_set, retrain_epochs, used_size=None):
        self.model.train()
        if used_size is None:
            train_loader = data.DataLoader(
                labeled_set, batch_size=10, shuffle=True, num_workers=2)
        else:
            indices = np.random.choice(
                len(labeled_set), used_size, replace=False)
            train_loader = data.DataLoader(
                labeled_set, batch_size=10,
                sampler=data.sampler.SubsetRandomSampler(indices),
                num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=True, num_workers=2)
        for epoch in range(retrain_epochs):
            self.train_step(train_loader, epoch)
            self.test(test_loader)

    def train_step(self, train_loader, epoch):
        for batch_idx, (x, target, w) in enumerate(train_loader):
            self.optimizer.zero_grad()
            x, target, w = (
                Variable(x).type(dtype),
                Variable(target).type(dtype),
                Variable(w).type(dtype))
            output = self.model(x)
            _, loss = self.compute_loss(output, target, w)
            loss.backward()
            self.optimizer.step()
        if (batch_idx+1) % 100 == 0:
            print(
                '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx+1, loss.data[0]))

    def basic_loss(self, fx):
        negative_logistic = nn.LogSigmoid()
        return -negative_logistic(fx)
        # sigmoid = nn.Sigmoid()
        # return sigmoid(-fx)

    def compute_loss(self, output, target, w=None):
        a = (self.pho_p - self.pho_n)/2
        b = (self.pho_p + self.pho_n)/2
        pho_y = a * target + b
        pho_ny = self.pho_p + self.pho_n - pho_y
        loss = self.basic_loss(target*output)
        loss = (1-pho_ny+pho_y) * loss
        if w is not None:
            assert w.shape == loss.shape
            loss *= w
        total_loss = torch.sum(loss)
        return loss, total_loss

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        for x, target in test_loader:
            x, target = Variable(x).type(dtype), Variable(target).type(dtype)
            output = self.model(x)
            pred = torch.sign(output)
            correct += torch.sum(pred.eq(target).float()).data[0]
        test_loss /= len(test_loader.dataset)
        print(
            'Test set: Accuracy: {}/{} ({:.2f}%)'.format(
                correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
