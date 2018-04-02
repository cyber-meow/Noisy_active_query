import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# import matplotlib.pyplot as plt
import numpy as np

import dataset
from active_query import RandomQuery, IWALQuery
from classifier import Classifier
from copy import deepcopy


# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Load and transform data
mnist = torchvision.datasets.MNIST(
    './MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    './MNIST', train=False, download=True, transform=transform)

test_data = mnist_test.test_data.numpy()
test_labels = mnist_test.test_labels.numpy()
used_idxs = np.logical_or(test_labels == 3, test_labels == 8)
test_labels = (test_labels-3)/2.5-1

test_set = data.TensorDataset(
    torch.from_numpy(test_data[used_idxs]).unsqueeze(1).float(),
    torch.from_numpy(test_labels[used_idxs]).unsqueeze(1).float())

train_data = mnist.train_data.numpy()
train_labels = mnist.train_labels.numpy()
used_idxs = np.logical_or(train_labels == 3, train_labels == 8)
train_labels = (train_labels-3)/2.5-1

train_data = train_data[used_idxs]
train_labels = train_labels[used_idxs]
init_weight = 300
init_size = 2000

train_data = torch.from_numpy(train_data).unsqueeze(1).float()
train_labels = torch.from_numpy(train_labels).unsqueeze(1).float()

pho_p = 0.5
pho_n = 0

unlabeled_set, labeled_set = dataset.datasets_initialization(
    train_data, train_labels, init_size,
    init_weight, pho_p=pho_p, pho_n=pho_n)
unlabeled_set_rand = deepcopy(unlabeled_set)
labeled_set_rand = deepcopy(labeled_set)


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


incr_times = 1
batch_size = 100
retrain_epochs = 20
learning_rate = 5e-3
query_batch_size = 10
# cls = Classifier(Net(pho_p=pho_p, pho_n=pho_n))
clss = [Classifier(Net(), pho_p=pho_p, pho_n=pho_n, lr=learning_rate)
        for _ in range(1)]
# clss = [Classifier(Net().cuda()) for _ in range(5)]
clss_rand = [deepcopy(cls) for cls in clss]
used_size = 900


for incr in range(incr_times):

    print('\nincr {}'.format(incr))

    '''
    print('\nActive Query'.format(incr))
    for i, cls in enumerate(clss):
        print('classifier {}'.format(i))
        cls.train(labeled_set, test_set, batch_size,
                  retrain_epochs, used_size)
    IWALQuery().query(unlabeled_set, labeled_set, query_batch_size, clss)
    used_size += query_batch_size - 1
    '''

    print('\nRandom Query'.format(incr))
    for i, cls in enumerate(clss_rand):
        print('classifier {}'.format(i))
        cls.train(labeled_set_rand, test_set, batch_size, retrain_epochs)
    RandomQuery().query(
        unlabeled_set_rand, labeled_set_rand, query_batch_size, init_weight)
