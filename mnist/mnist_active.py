import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import argparse
# import matplotlib.pyplot as plt
import numpy as np
# from sklearn.decomposition import PCA
from copy import deepcopy
from collections import OrderedDict

import dataset
import settings
from active_query import RandomQuery, IWALQuery
from classifier import Classifier, majority_vote
from mnist.basics import Net, Linear


pho_p = 0.4
pho_n = 0

batch_size = 40
learning_rate = 5e-4
weight_decay = 1e-2

convex_epochs = 10
retrain_epochs = 120
test_on_train = False

num_clss = 1
init_size = 2600

used_size = 550
incr_times = 0
query_batch_size = 40
reduced_sample_size = 4

init_weight = 1
weight_ratio = 2

use_CNN = False
kcenter = False

params = OrderedDict([
    ('kcenter', kcenter),
    ('use_CNN', use_CNN),
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


parser = argparse.ArgumentParser(description='MNIST noise active learning')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--no-active', action='store_true', default=False,
                    help='disables active learning')

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
    'datasets/MNIST', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(
    'datasets/MNIST', train=False, download=True, transform=transform)


train_data = mnist.train_data.numpy()
train_labels = mnist.train_labels.numpy()
used_idxs = np.logical_or(train_labels == 3, train_labels == 8)
train_labels = (train_labels-3)/2.5-1
# used_idxs = np.logical_or(train_labels == 7, train_labels == 9)
# train_labels = train_labels-8

# pca = PCA(n_components=n_pca_components)
# train_data = pca.fit_transform(train_data.reshape(-1, 784))

train_data = train_data[used_idxs]
train_labels = train_labels[used_idxs]

train_data = torch.from_numpy(train_data).unsqueeze(1).float()
train_labels = torch.from_numpy(train_labels).unsqueeze(1).float()
train_labels = dataset.label_corruption(train_labels, pho_p, pho_n)

data_init = (dataset.datasets_initialization_kcenter
             if kcenter
             else dataset.datasets_initialization)

unlabeled_set, labeled_set = data_init(
    train_data, train_labels, init_size, init_weight)
unlabeled_set_rand = deepcopy(unlabeled_set)
labeled_set_rand = deepcopy(labeled_set)


test_data = mnist_test.test_data.numpy()
test_labels = mnist_test.test_labels.numpy()
used_idxs = np.logical_or(test_labels == 3, test_labels == 8)
test_labels = (test_labels-3)/2.5-1
# used_idxs = np.logical_or(test_labels == 7, test_labels == 9)
# test_labels = test_labels-8

# test_data = pca.transform(test_data.reshape(-1, 784))

test_set = data.TensorDataset(
    torch.from_numpy(test_data[used_idxs]).unsqueeze(1).float(),
    torch.from_numpy(test_labels[used_idxs]).unsqueeze(1).float())


def create_new_classifier():
    if use_CNN:
        model = Net().cuda() if args.cuda else Net()
    else:
        model = Linear().cuda() if args.cuda else Linear()
    cls = Classifier(
            model,
            pho_p=pho_p,
            pho_n=pho_n,
            lr=learning_rate,
            weight_decay=weight_decay)
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
