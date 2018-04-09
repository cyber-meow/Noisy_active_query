import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import settings
from classifier import Classifier
from dataset import WeightedTensorDataset


init_weight = 1

init_p_size = 2500
init_n_size = 2500
init_p_un_size = 0
init_n_un_size = 0
# uncertainty_pool = 3500
uncertainty_pool_p = 1000
uncertainty_pool_n = 1000

pho_p = 0
pho_n = 0

batch_size = 200
num_clss = 1
learning_rate = 5e-3
test_on_train = False

retrain_epochs = 20
convex_epochs = 5


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
train_labels = torch.from_numpy(y_train.values).unsqueeze(1).float()
training_set = WeightedTensorDataset(
    train_data, train_labels, init_weight * torch.ones(len(train_data), 1))

test_set = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test.values).float(),
    torch.from_numpy(y_test.values).unsqueeze(1).float())


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


# model = Net().cuda() if args.cuda else Net()
model = Linear().cuda() if args.cuda else Net()
cls = Classifier(model)
cls.train(training_set, test_set, batch_size, 4, 1)
output = cls.model(Variable(train_data).type(settings.dtype)).cpu()
probs = F.sigmoid(output).data.numpy().reshape(-1)
sorted_margin = torch.from_numpy(np.argsort(np.abs(probs-0.5)))


p_sorted_margin = []
n_sorted_margin = []

for i in sorted_margin:
    if train_labels[i][0] == 1:
        p_sorted_margin.append(i)
    else:
        n_sorted_margin.append(i)

tmp_labels = train_labels.numpy().reshape(-1)
p_idxs = tmp_labels == 1
n_idxs = tmp_labels == -1

# un_idxs = np.zeros(len(train_data))
# un_idxs[sorted_margin[:uncertainty_pool]] = True

# p_un = np.argwhere(np.logical_and(p_idxs, un_idxs)).reshape(-1)
p_un = p_sorted_margin[:uncertainty_pool_p]
drawn = np.random.choice(p_un, init_p_un_size, replace=False)

# n_un = np.argwhere(np.logical_and(n_idxs, un_idxs)).reshape(-1)
n_un = n_sorted_margin[:uncertainty_pool_n]
drawn2 = np.random.choice(n_un, init_n_un_size, replace=False)

dr_idxs = np.zeros(len(train_data))
dr_idxs[drawn] = True
dr_idxs[drawn2] = True

p_rest = np.argwhere(
    np.logical_and(p_idxs, np.logical_not(dr_idxs))).reshape(-1)
n_rest = np.argwhere(
    np.logical_and(n_idxs, np.logical_not(dr_idxs))).reshape(-1)

drawn3 = np.random.choice(p_rest, init_p_size, replace=False)
drawn4 = np.random.choice(n_rest, init_n_size, replace=False)

drawn = torch.from_numpy(
    np.concatenate([drawn, drawn2, drawn3, drawn4]))


given_data = train_data[drawn]
given_labels = train_labels[drawn]

for i, label in enumerate(given_labels):
    assert label[0] == 1 or label[0] == -1
    if label[0] == 1 and np.random.random() < pho_p:
        # print('flip +1')
        given_labels[i] = -1
    elif np.random.random() < pho_n:
        # print('flip -1')
        given_labels[i] = 1

labeled_set = WeightedTensorDataset(
    given_data, given_labels, torch.ones(len(given_data), 1))
print(len(labeled_set))


def create_new_classifier():
    model = Net().cuda() if args.cuda else Net()
    # model = Linear().cuda() if args.cuda else Linear()
    cls = Classifier(
            model,
            pho_p=pho_p,
            pho_n=pho_n,
            lr=learning_rate)
    return cls


clss = [create_new_classifier() for _ in range(num_clss)]


for i, cls in enumerate(clss):
    print('classifier {}'.format(i))
    cls.train(
        labeled_set, test_set, batch_size,
        retrain_epochs, convex_epochs, test_on_train=test_on_train)
