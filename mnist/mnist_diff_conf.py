import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

import argparse
# import matplotlib.pyplot as plt
import numpy as np
import pickle
from copy import deepcopy
from collections import OrderedDict

import dataset
import settings
from active_query import RandomQuery, DisagreementQuery
from classifier import Classifier
from mnist.basics import Net, Linear


pho_p = 0.4
pho_n = 0.4

batch_size = 40
learning_rate = 5e-4
weight_decay = 5e-2

init_epochs = 100
init_convex_epochs = 0
retrain_convex_epochs = 0
retrain_epochs = 40
test_on_train = False

init_size = 100
query_times = 5
query_batch_size = 20

neigh_size = 5
init_weight = 1

use_CNN = True
kcenter = False

params = OrderedDict([
    ('kcenter', kcenter),
    ('use_CNN', use_CNN),
    ('\npho_p', pho_p),
    ('pho_n', pho_n),
    ('\nbatch_size', batch_size),
    ('learning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('\ninit_convex_epochs', init_convex_epochs),
    ('retrain_convex_epochs', retrain_convex_epochs),
    ('retrain_epochs', retrain_epochs),
    ('\ninit_size', init_size),
    ('query_times', query_times),
    ('query_batch_size', query_batch_size),
    ('\ninit_weight', init_weight),
])

for key, value in params.items():
    print('{}: {}'.format(key, value))
print('')


parser = argparse.ArgumentParser(description='MNIST noise active learning')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--no-active', action='store_true', default=False,
                    help='disables active learning')

parser.add_argument('--save', help='save the initialization to some file')
parser.add_argument('--load', help='load the initialization from some file')

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

train_data = train_data[used_idxs]
train_labels = train_labels[used_idxs]

train_data = torch.from_numpy(train_data).unsqueeze(1).float()
train_labels = torch.from_numpy(train_labels).unsqueeze(1).float()

if args.load is not None:
    unlabeled_set, labeled_set, cls = pickle.load(open(args.load, 'rb'))
    if args.cuda:
        cls.model = cls.model.cuda()
    cls.use_best = True

else:
    data_init = (dataset.datasets_initialization_kcenter
                 if kcenter
                 else dataset.datasets_initialization)

    unlabeled_set, labeled_set = data_init(
        train_data, train_labels, init_size, init_weight, pho_p, pho_n)

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
            weight_decay=weight_decay,
            use_best=True)
    return cls


if not args.load:
    cls = create_new_classifier()
cls_rand = deepcopy(cls)
cls_end = deepcopy(cls)
cls_rand_end = deepcopy(cls)

if args.save is not None:
    pickle.dump((unlabeled_set, labeled_set, cls), open(args.save, 'wb'))


for query in range(query_times+1):

    print('\nQuery {}'.format(query))
    convex_epochs = (init_convex_epochs
                     if query == 0
                     else retrain_convex_epochs)
    num_epochs = init_epochs if query == 0 else retrain_epochs

    if not args.no_active:

        print('\nActive Query'.format(query))

        print('\nClassifier Main')
        cls.train(labeled_set, test_set, batch_size,
                  num_epochs, convex_epochs,
                  test_on_train=test_on_train)

        labeled_set.is_used_tensor[:] = 1
        labeled_set.drop_and(cls, fraction=1)
        # labeled_set.drop_local_inconsitent(fraction=0.5)

        flipped_set = deepcopy(labeled_set)
        flipped_set.is_used_tensor[:] = 1
        flipped_set.drop_and(cls, fraction=1.5)
        # flipped_set.drop_local_inconsitent(fraction=1)

        print('\nClassifier Relaxed')
        new_cls = deepcopy(cls)
        new_cls.train(flipped_set, test_set, batch_size, retrain_epochs,
                      convex_epochs, test_on_train=test_on_train)

        labeled_set.weight_tensor[:] *= 1/2
        DisagreementQuery().query(
            unlabeled_set, labeled_set, query_batch_size,
            [cls, new_cls], init_weight)

    print('\nRandom Query'.format(query))

    cls_rand.train(
        labeled_set_rand, test_set, batch_size,
        num_epochs, convex_epochs, test_on_train=test_on_train)

    labeled_set_rand.is_used_tensor[:] = 1
    labeled_set_rand.drop_and(cls_rand, fraction=1.2)
    # labeled_set_rand.drop_local_inconsitent(fraction=1)

    labeled_set_rand.weight_tensor[:] *= 1/2
    RandomQuery().query(
        unlabeled_set_rand, labeled_set_rand, query_batch_size, init_weight)


if query_times > 0:

    print('\n\nTrain new classifier on selected points')

    cls = cls_end
    cls_rand = cls_rand_end

    if not args.no_active:
        print('\nActively Selected Points')
        labeled_set.weight_tensor[:] = 1
        cls.train(
            labeled_set, test_set, batch_size,
            init_epochs, init_convex_epochs,
            test_on_train=test_on_train)

    print('\nRandomly Selected Points')
    labeled_set_rand.weight_tensor[:] = 1
    cls_rand.train(
        labeled_set_rand, test_set, batch_size,
        init_epochs, init_convex_epochs,
        test_on_train=test_on_train)
