import sys

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from copy import deepcopy
import settings


class DoubleClassifier(object):

    def __init__(self, model, lr=5e-3, weight_decay=1e-2,
                 pho_p=0, pho_n=0, use_best=False):
        self.model = model
        self.model2 = deepcopy(model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.pho_p = pho_p
        self.pho_n = pho_n

        self.best_model = None
        self.best_model2 = None
        self.use_best = use_best

        self.test_accuracies = []
        self.train_accuracies = []
        self.train_clean_accuracies = []
        self.train_noise_accuracies = []

        self.test_accuracies2 = []
        self.train_accuracies2 = []
        self.train_clean_accuracies2 = []
        self.train_noise_accuracies2 = []

        self.high_loss_noise_fractions = []
        self.high_loss_noise_fractions2 = []

        self.noise_confs = []
        self.clean_confs = []
        self.confs = []

        self.noise_confs2 = []
        self.clean_confs2 = []
        self.confs2 = []

        self.dis_nums = []
        self.dis_noise_nums = []

        self.init_optimizer()

    def init_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer2 = optim.Adam(
            self.model2.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)

    def train(self, labeled_set, test_set,
              batch_size, retrain_epochs,
              convex_epochs=None, test_interval=1,
              print_interval=1, test_on_train=False):

        self.init_optimizer()
        self.best_accuracy = 0
        self.best_accuracy2 = 0

        self.noise = (torch.sign(labeled_set.target_tensor)
                      != labeled_set.label_tensor
                      ).numpy().reshape(-1).astype(bool)
        self.clean = np.logical_not(self.noise)

        self.correct_accumulation = np.zeros(len(labeled_set))
        self.noise_correct_accumulation = np.zeros(self.noise.sum())
        self.clean_correct_accumulation = np.zeros(self.clean.sum())

        self.correct_accumulation2 = np.zeros(len(labeled_set))
        self.noise_correct_accumulation2 = np.zeros(self.noise.sum())
        self.clean_correct_accumulation2 = np.zeros(self.clean.sum())

        if test_on_train:
            self.test_on_train_noise(labeled_set)
        self.test(test_set, 'Test', True)

        train_loader = data.DataLoader(
            labeled_set, batch_size=batch_size,
            shuffle=True, num_workers=2)

        for epoch in range(retrain_epochs):

            if convex_epochs is not None and epoch >= convex_epochs:
                self.train_step(train_loader, epoch, convex_loss=False)
            else:
                self.train_step(train_loader, epoch, convex_loss=True)

            if (epoch+1) % test_interval == 0 or epoch+1 == retrain_epochs:

                to_print = (epoch+1) % print_interval == 0
                if to_print:
                    sys.stdout.write('Epoch: {} '.format(epoch))
                if test_on_train:
                    self.test_on_train_noise(labeled_set)
                self.test(test_set, 'Test', to_print)
                self.find_high_loss_samples(labeled_set, to_print)

        if self.best_model is not None and self.use_best:
            self.model = self.best_model
        if self.best_model2 is not None and self.use_best:
            self.model2 = self.best_model2
        self.test(test_set, 'Test')
        self.test_accuracies.pop()
        self.test_accuracies2.pop()

    def train_step(self, train_loader, epoch,
                   convex_loss=True):

        self.model.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            x, target = (
                Variable(x).type(settings.dtype),
                Variable(target).type(settings.dtype))
            output = self.model(x)
            _, loss = self.compute_loss(output, target, convex_loss)
            loss.backward()
            self.optimizer.step()

        self.model2.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            self.optimizer2.zero_grad()
            x, target = (
                Variable(x).type(settings.dtype),
                Variable(target).type(settings.dtype))
            output = self.model2(x)
            _, loss = self.compute_loss(output, -target, convex_loss)
            loss.backward()
            self.optimizer2.step()

    def basic_loss(self, fx, convex_loss=True):
        if convex_loss:
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        else:
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)

    def compute_loss(self, output, target, convex_loss=True):
        loss = self.basic_loss(torch.sign(target)*output, convex_loss)
        loss = torch.abs(target) * loss
        total_loss = torch.sum(loss)
        return loss, total_loss

    def find_high_loss_samples(self, labeled_set, to_print=True):
        noise = self.noise
        clean = self.clean

        self.model.eval()
        x = Variable(labeled_set.data_tensor).type(settings.dtype)
        target = Variable(
            torch.sign(labeled_set.target_tensor)).type(settings.dtype)
        noise_rate = (self.pho_p + self.pho_n)/2
        output = self.model(x)
        prob = self.basic_loss(
            -output*target, False).data.cpu().numpy().reshape(-1)
        prob_th = np.percentile(prob, noise_rate*100)
        high_loss_noise_fraction = np.sum(
            np.logical_and(noise, prob < prob_th))/np.sum(prob < prob_th)
        self.high_loss_noise_fractions.append(high_loss_noise_fraction*100)
        self.confs.append(np.mean(prob)*100)
        self.noise_confs.append(np.mean(prob[noise])*100)
        self.clean_confs.append(np.mean(prob[clean])*100)

        self.model2.eval()
        output = self.model2(x)
        prob = self.basic_loss(
            output*target, False).data.cpu().numpy().reshape(-1)
        prob_th = np.percentile(prob, noise_rate*100)
        high_loss_noise_fraction = np.sum(
            np.logical_and(noise, prob < prob_th))/np.sum(prob < prob_th)
        self.high_loss_noise_fractions2.append(high_loss_noise_fraction*100)
        self.confs2.append(np.mean(prob)*100)
        self.noise_confs2.append(np.mean(prob[noise])*100)
        self.clean_confs2.append(np.mean(prob[clean])*100)

    def test(self, test_set, set_name, to_print=True):

        self.model.eval()
        x = Variable(test_set.data_tensor).type(settings.dtype)
        target = Variable(
            torch.sign(test_set.target_tensor)).type(settings.dtype)
        output = self.model(x)
        pred = torch.sign(output)
        correct = torch.sum(pred.eq(target).float()).data[0]
        accuracy = 100 * correct/len(test_set)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = deepcopy(self.model)
        self.test_accuracies.append(accuracy)
        if to_print:
            print('{} set: Accuracy: {}/{} ({:.2f}%)'.format(
                set_name, correct, len(test_set), accuracy))

        self.model2.eval()
        output = self.model2(x)
        pred = -torch.sign(output)
        correct = torch.sum(pred.eq(target).float()).data[0]
        accuracy = 100 * correct/len(test_set)
        if accuracy > self.best_accuracy2:
            self.best_accuracy2 = accuracy
            self.best_model2 = deepcopy(self.model2)
        self.test_accuracies2.append(accuracy)
        if to_print:
            print('Model 2: {} set: Accuracy: {}/{} ({:.2f}%)'.format(
                set_name, correct, len(test_set), accuracy))

    def test_on_train_noise(self, training_set):
        noise = self.noise
        clean = self.clean

        self.model.eval()
        x = Variable(training_set.data_tensor).type(settings.dtype)
        target = Variable(torch.sign(training_set.target_tensor))
        output = self.model(x).cpu()
        pred = torch.sign(output)
        corrects = pred.eq(target).data.numpy().reshape(-1)
        self.correct_accumulation += corrects
        self.noise_correct_accumulation += corrects[noise]
        self.clean_correct_accumulation += corrects[clean]
        accuracy = 100 * np.mean(corrects)
        noise_accuracy = 100 * np.mean(corrects[noise])
        clean_accurary = 100 * np.mean(corrects[clean])
        self.train_accuracies.append(accuracy)
        self.train_noise_accuracies.append(noise_accuracy)
        self.train_clean_accuracies.append(clean_accurary)

        self.model2.eval()
        output = self.model2(x).cpu()
        pred2 = -torch.sign(output)
        corrects = pred2.eq(target).data.numpy().reshape(-1)
        self.correct_accumulation2 += corrects
        self.noise_correct_accumulation2 += corrects[noise]
        self.clean_correct_accumulation2 += corrects[clean]
        accuracy = 100 * np.mean(corrects)
        noise_accuracy = 100 * np.mean(corrects[noise])
        clean_accurary = 100 * np.mean(corrects[clean])
        self.train_accuracies2.append(accuracy)
        self.train_noise_accuracies2.append(noise_accuracy)
        self.train_clean_accuracies2.append(clean_accurary)

        disagree = (pred != pred2).data.numpy().reshape(-1)
        disagree_noise = np.logical_and(noise, disagree)
        print(np.sum(disagree_noise))
        self.dis_nums.append(np.sum(disagree))
        self.dis_noise_nums.append(np.sum(disagree_noise))
