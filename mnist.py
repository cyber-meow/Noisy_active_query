import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import argparse
from copy import deepcopy
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import dataset
import settings
from active_query import RandomQuery, IWALQuery
from classifier import Classifier


init_weight = 30
init_size = 500

pho_p = 0.5
pho_n = 0

batch_size = 100
num_clss = 2
learning_rate = 5e-3
incr_times = 2
test_on_train = False

retrain_epochs = 4
convex_epochs = 2
query_batch_size = 100
reduced_sample_size = 5
used_size = 450

n_pca_components = 50


parser = argparse.ArgumentParser(description='MNIST noise active learning')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    settings.dtype = torch.cuda.FloatTensor


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './MNIST', train=False, download=True, transform=transform)


train_data = mnist.train_data.numpy()
train_labels = mnist.train_labels.numpy()
used_idxs = np.logical_or(train_labels == 3, train_labels == 8)
train_labels = (train_labels-3)/2.5-1
# used_idxs = np.logical_or(train_labels == 7, train_labels == 9)
# train_labels = train_labels-8

pca = PCA(n_components=n_pca_components)
train_data = pca.fit_transform(train_data.reshape(-1, 784))

train_data = train_data[used_idxs]
train_labels = train_labels[used_idxs]

train_data = torch.from_numpy(train_data).unsqueeze(1).float()
train_labels = torch.from_numpy(train_labels).unsqueeze(1).float()

unlabeled_set, labeled_set = dataset.datasets_initialization(
    train_data, train_labels, init_size,
    init_weight, pho_p=pho_p, pho_n=pho_n)
unlabeled_set_rand = deepcopy(unlabeled_set)
labeled_set_rand = deepcopy(labeled_set)


test_data = mnist_test.test_data.numpy()
test_labels = mnist_test.test_labels.numpy()
used_idxs = np.logical_or(test_labels == 3, test_labels == 8)
test_labels = (test_labels-3)/2.5-1
# used_idxs = np.logical_or(test_labels == 7, test_labels == 9)
# test_labels = test_labels-8

test_data = pca.transform(test_data.reshape(-1, 784))

test_set = data.TensorDataset(
    torch.from_numpy(test_data[used_idxs]).unsqueeze(1).float(),
    torch.from_numpy(test_labels[used_idxs]).unsqueeze(1).float())


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.fc1 = nn.Linear(4*4*10, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Linear(n_pca_components, 1)

    def forward(self, x):
        y_pred = self.linear(x.view(-1, n_pca_components))
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


for incr in range(incr_times+1):

    print('\nincr {}'.format(incr))

    '''
    print('\nActive Query'.format(incr))
    for i, cls in enumerate(clss):
        print('classifier {}'.format(i))
        cls.train(labeled_set, test_set, batch_size,
                  retrain_epochs, convex_epochs, used_size, test_on_train)
    IWALQuery().query(unlabeled_set, labeled_set, query_batch_size, clss)
    used_size += query_batch_size - reduced_sample_size
    '''

    print('\nRandom Query'.format(incr))
    for i, cls in enumerate(clss_rand):
        print('classifier {}'.format(i))
        cls.train(
            labeled_set_rand, test_set, batch_size,
            retrain_epochs, convex_epochs, test_on_train=test_on_train)
    RandomQuery().query(
        unlabeled_set_rand, labeled_set_rand, query_batch_size, init_weight)
