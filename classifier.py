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
        self.threshold = 1
        self.best_accuracy = 0
        self.use_logistic_threshold = 75
        self.last_accuracy = 50

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

            # output = self.model(
            #     Variable(labeled_set.data_tensor).type(settings.dtype))
            # fxs = (output.data.cpu()
            #        * labeled_set.target_tensor).numpy().reshape(-1)
            # negative_fxs = fxs[fxs <= 0]
            # tmp = (1 - np.array(negative_fxs))/2
            # self.threshold = np.percentile(tmp, 30).item()
            # print(self.threshold)
            # print(np.median(tmp))

            if convex_epochs is not None and epoch >= convex_epochs:
                if self.last_accuracy < self.use_logistic_threshold:
                    print('use logistic exceptionally')
                    self.train_step(train_loader, epoch)
                    # self.train_step(train_loader, epoch, convex_loss=False)
                else:
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
            # return torch.clamp((1-fx)/2, 0, self.threshold)

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
        self.last_accuracy = 100 * correct / len(test_loader.dataset)
        self.best_accuracy = max(self.best_accuracy, self.last_accuracy)
        self.use_logistic_threshold = max(
            self.use_logistic_threshold, self.best_accuracy-5)
        print(
            '{} set: Accuracy: {}/{} ({:.2f}%)'.format(
                set_name, correct, len(test_loader.dataset),
                self.last_accuracy))


def majority_vote(clss, test_set):
    pred = torch.zeros_like(test_set.target_tensor)
    for cls in clss:
        cls.model.eval()
        output = cls.model(
            Variable(test_set.data_tensor).type(settings.dtype))
        pred += torch.sign(output).data.cpu()
    pred = torch.sign(pred)
    correct = torch.sum(pred.eq(test_set.target_tensor).float())
    accuracy = 100 * correct / len(test_set)
    print(
        'Majority vote: Accuracy: {}/{} ({:.2f}%)'.format(
            correct, len(test_set), accuracy))
