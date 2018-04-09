import torch
import torch.nn as nn
import torch.utils.data

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy

import dataset
import settings
from active_query import RandomQuery, IWALQuery
from classifier import Classifier, majority_vote


init_weight = 1
weight_ratio = 2
init_size = 100

pho_p = 0.1
pho_n = 0.1

batch_size = 50
num_clss = 1
learning_rate = 5e-3
incr_times = 0
test_on_train = False

retrain_epochs = 120
convex_epochs = 10
query_batch_size = 40
reduced_sample_size = 4
used_size = 90


parser = argparse.ArgumentParser(description='MNIST noise active learning')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    settings.dtype = torch.cuda.FloatTensor


data = pd.read_csv("datasets/mushrooms/mushrooms.csv")

target = 'class'
labels = data[target]

features = data.drop(target, axis=1)

categorical = features.columns
features = pd.concat(
    [features, pd.get_dummies(features[categorical])], axis=1)
features.drop(categorical, axis=1, inplace=True)

labels = pd.get_dummies(labels)['e']
labels = labels.astype(int)*2-1

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=0)

train_data = torch.from_numpy(X_train.values).float()
# m = torch.mean(train_data, 1, keepdim=True)
# std = torch.std(train_data, 1, keepdim=True)
# train_data = (train_data-m)/std
train_labels = torch.from_numpy(y_train.values).unsqueeze(1).float()

unlabeled_set, labeled_set = dataset.datasets_initialization(
    train_data, train_labels, init_size,
    init_weight, pho_p=pho_p, pho_n=pho_n)
unlabeled_set_rand = deepcopy(unlabeled_set)
labeled_set_rand = deepcopy(labeled_set)

test_data = torch.from_numpy(X_test.values).float()
# m = torch.mean(test_data, 1, keepdim=True)
# std = torch.std(test_data, 1, keepdim=True)
# test_data = (test_data-m)/std
test_set = torch.utils.data.TensorDataset(
    test_data, torch.from_numpy(y_test.values).unsqueeze(1).float())


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(117, 50)
        self.linear2 = nn.Linear(50, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Linear(117, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def create_new_classifier():
    # model = Net().cuda() if args.cuda else Net()
    model = Linear().cuda() if args.cuda else Linear()
    cls = Classifier(
            model,
            pho_p=pho_p,
            pho_n=pho_n,
            lr=learning_rate)
    return cls


clss = [create_new_classifier() for _ in range(num_clss)]
clss_rand = [deepcopy(cls) for cls in clss]
IWALQuery = IWALQuery()


for incr in range(incr_times+1):

    print('\nincr {}'.format(incr))

    '''
    print('\nActive Query'.format(incr))
    for i, cls in enumerate(clss):
        print('classifier {}'.format(i))
        cls.train(labeled_set, test_set, batch_size,
                  retrain_epochs, convex_epochs, used_size, test_on_train)
    drawn_number = IWALQuery.query(
        unlabeled_set, labeled_set, query_batch_size, clss, weight_ratio)
    used_size += drawn_number - reduced_sample_size
    majority_vote(clss, test_set)
    '''

    print('\nRandom Query'.format(incr))
    for i, cls in enumerate(clss_rand):
        print('classifier {}'.format(i))
        cls.train(
            labeled_set_rand, test_set, batch_size,
            retrain_epochs, convex_epochs, test_on_train=test_on_train)
    RandomQuery().query(
        unlabeled_set_rand, labeled_set_rand, query_batch_size, init_weight)
    majority_vote(clss_rand, test_set)
