import torch
import torch.nn as nn
import torch.utils.data

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import deepcopy
from collections import OrderedDict

import dataset
import settings
from active_query import RandomQuery, IWALQuery
from classifier import Classifier, majority_vote


pho_p = 0.1
pho_n = 0.1

batch_size = 50
learning_rate = 5e-3
weight_decay = 1e-2

convex_epochs = 10
retrain_epochs = 120
test_on_train = False

num_clss = 5
init_size = 80

used_size = 75
incr_times = 8
query_batch_size = 15
reduced_sample_size = 2

init_weight = 1
weight_ratio = 2

kcenter = False

params = OrderedDict([
    ('kcenter', kcenter),
    ('\npho_p', pho_p),
    ('pho_n', pho_n),
    ('\nbatch_size', batch_size),
    ('learning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('\nconvex_epochs', convex_epochs),
    ('retrain_epochs', retrain_epochs),
    ('\nnum_clss', num_clss),
    ('init_size', init_size),
    ('used_size', used_size),
    ('incr_times', incr_times),
    ('query_batch_size', query_batch_size),
    ('reduced_sample_size', reduced_sample_size),
    ('\ninit_weight', init_weight),
    ('weight_ratio', weight_ratio),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(
    description='UCI mushrooms noise active learning')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-active', action='store_true', default=False,
                    help='disables active learning')
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

data_init = (dataset.datasets_initialization_kcenter
             if kcenter
             else dataset.datasets_initialization)

unlabeled_set, labeled_set = data_init(
    train_data, train_labels, init_size, init_weight, pho_p, pho_n)
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

    if not args.no_active:
        print('\nActive Query'.format(incr))
        for i, cls in enumerate(clss):
            print('\nclassifier {}'.format(i))
            cls.train(labeled_set, test_set, batch_size,
                      retrain_epochs, convex_epochs, used_size,
                      test_on_train=test_on_train)
        selected = IWALQuery.query(
            unlabeled_set, labeled_set, query_batch_size, clss, weight_ratio)
        used_size += len(selected[0]) - reduced_sample_size
        majority_vote(clss, test_set)

    print('\nRandom Query'.format(incr))
    for i, cls in enumerate(clss_rand):
        print('\nclassifier {}'.format(i))
        cls.train(
            labeled_set_rand, test_set, batch_size,
            retrain_epochs, convex_epochs, test_on_train=test_on_train)
    RandomQuery().query(
        unlabeled_set_rand, labeled_set_rand, query_batch_size, init_weight)
    if num_clss > 1:
        majority_vote(clss_rand, test_set)


if incr_times > 0:

    print('\n\nTrain new classifier on selected points')

    cls = create_new_classifier()
    cls_rand = deepcopy(cls)

    if not args.no_active:
        print('\nActively Selected Points')
        cls.train(
            labeled_set, test_set, batch_size,
            retrain_epochs*2, convex_epochs, test_on_train=test_on_train)

    print('\nRandomly Selected Points')
    cls_rand.train(
        labeled_set_rand, test_set, batch_size,
        retrain_epochs*2, convex_epochs, test_on_train=test_on_train)
