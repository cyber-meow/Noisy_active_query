import sys

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from copy import deepcopy
import settings


class Classifier(object):

    def __init__(self, model, pho_p=0, pho_n=0, lr=5e-3, weight_decay=1e-2):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.pho_p = pho_p
        self.pho_n = pho_n
        self.threshold = 1
        self.init_optimizer()
        self.smallest_conf = 0
        self.critic_model = None
        self.test_accuracies = []
        self.train_accuracies = []
        self.high_loss_fractions = []
        self.critic_confs = []

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, labeled_set, test_set,
              batch_size, retrain_epochs,
              convex_epochs=None, used_size=None,
              test_interval=1, print_interval=1, test_on_train=False):

        self.init_optimizer()

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

        for epoch in range(retrain_epochs):

            if epoch == convex_epochs:
                self.smallest_conf = 100

            if convex_epochs is not None and epoch >= convex_epochs:
                if (self.test_accuracies != []
                        and self.test_accuracies[-1] < 75):
                    print('Use logistic exceptionally')
                    self.train_step(train_loader, epoch)
                    # self.train_step(train_loader, epoch, convex_loss=False)
                else:
                    self.train_step(train_loader, epoch, convex_loss=False)
            else:
                self.train_step(train_loader, epoch)

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
            self.test(labeled_set, 'Train')
            self.train_accuracies.pop()
        self.test(test_set, 'Test')
        self.test_accuracies.pop()

    def train_step(self, train_loader, epoch, convex_loss=True):
        self.model.train()
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
                print('==>>> Batch index: {}, Train Loss: {:.6f}'.format(
                        batch_idx+1, total_loss))

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
        # loss_ny = self.basic_loss(-targets*outputs, convex_loss)
        loss = (1-pho_ny+pho_y) * loss
        if w is not None:
            assert w.shape == loss.shape
            loss *= w
        total_loss = torch.sum(loss)
        return loss, total_loss

    def find_high_loss_samples(self, labeled_set):
        self.model.eval()
        x = Variable(labeled_set.data_tensor).type(settings.dtype)
        target = Variable(labeled_set.target_tensor).type(settings.dtype)
        output = self.model(x)
        prob = self.basic_loss(-output*target, False).data.numpy().reshape(-1)
        high_loss_fraction = np.sum(prob < 0.4)/len(prob)*100
        small_conf_number = int(len(prob)*(self.pho_p+self.pho_n)/2)
        critic_conf = np.mean(np.sort(prob)[:small_conf_number])*100
        if critic_conf < self.smallest_conf:
            self.smallest_conf = critic_conf
            self.critic_model = deepcopy(self.model)
        self.critic_confs.append(critic_conf)
        self.high_loss_fractions.append(high_loss_fraction)
        # print('High loss samples: {} %'.format(high_loss_fraction))

    def test(self, test_set, set_name, to_print=True):
        self.model.eval()
        x = Variable(test_set.data_tensor).type(settings.dtype)
        target = Variable(test_set.target_tensor).type(settings.dtype)
        output = self.model(x)
        pred = torch.sign(output)
        correct = torch.sum(pred.eq(target).float()).data[0]
        accuracy = 100 * correct/len(test_set)
        if set_name == 'Test':
            self.test_accuracies.append(accuracy)
        if set_name == 'Train':
            self.train_accuracies.append(accuracy)
        if to_print:
            print('{} set: Accuracy: {}/{} ({:.2f}%)'.format(
                set_name, correct, len(test_set), accuracy))


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
