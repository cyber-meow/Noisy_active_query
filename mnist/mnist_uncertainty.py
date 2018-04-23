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
from active_query import RandomQuery, UncertaintyQuery
from active_query import DisagreementQuery, ClsDisagreementQuery
# from active_query import HeuristicRelabel
from classifier import Classifier
from mnist.basics import Net, Linear


pho_p = 0.3
pho_n = 0.3

batch_size = 40
learning_rate = 5e-4
weight_decay = 5e-2

init_convex_epochs = 20
init_epochs = 100
retrain_convex_epochs = 0
retrain_epochs = 40
test_on_train = False

init_size = 80

incr_times = 5
query_batch_size = 40
incr_pool_size = 700

neigh_size = 5

init_weight = 1
weight_ratio = 1

use_CNN = True
kcenter = False
uncertainty = False
compare_with_perfect = True
local_noise_drop = False
cls_loss_drop = True
true_noise_drop = False

params = OrderedDict([
    ('kcenter', kcenter),
    ('use_CNN', use_CNN),
    ('uncertainty', uncertainty),
    ('compare_with_perfect', compare_with_perfect),
    ('\nlocal_noise_drop', local_noise_drop),
    ('cls_loss_drop', cls_loss_drop),
    ('true_noise_drop', true_noise_drop),
    ('\npho_p', pho_p),
    ('pho_n', pho_n),
    ('\nbatch_size', batch_size),
    ('learning_rate', learning_rate),
    ('weight_decay', weight_decay),
    ('\ninit_epochs', init_epochs),
    ('init_convex_epochs', init_convex_epochs),
    ('retrain_convex_epochs', retrain_convex_epochs),
    ('retrain_epochs', retrain_epochs),
    ('\ninit_size', init_size),
    ('incr_times', incr_times),
    ('query_batch_size', query_batch_size),
    ('incr_pool_size', incr_pool_size),
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

data_init = (dataset.datasets_initialization_kcenter
             if kcenter
             else dataset.datasets_initialization)

unlabeled_set, labeled_set = data_init(
    train_data, train_labels, init_size, init_weight, pho_p, pho_n)
unlabeled_set_rand = deepcopy(unlabeled_set)
labeled_set_rand = deepcopy(labeled_set)

training_set = data.TensorDataset(train_data, train_labels)

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


if not uncertainty and compare_with_perfect:
    model = Net().cuda() if args.cuda else Net()
    perfect_cls = Classifier(model, lr=5e-3, weight_decay=0)
    perfect_cls.train(training_set, test_set, 200, 4, 1)


cls = create_new_classifier()
cls_rand = deepcopy(cls)
cls_end = deepcopy(cls)
cls_rand_end = deepcopy(cls)


for incr in range(incr_times):

    print('\nincr {}'.format(incr))
    convex_epochs = (init_convex_epochs
                     if incr == 0
                     else retrain_convex_epochs)
    num_epochs = init_epochs if incr == 0 else retrain_epochs

    if not args.no_active:
        print('\nActive Query'.format(incr))
        labeled_set.is_used_tensor[:] = 1
        if local_noise_drop:
            labeled_set.drop_local_inconsitent()
        cls.train(labeled_set, test_set, batch_size, num_epochs,
                  convex_epochs, test_on_train=test_on_train)

        if incr == 0:
            cls2 = deepcopy(cls)
        else:
            lset = deepcopy(labeled_set)
            lset.weight_tensor[:] = 1
            print('')
            cls2.train(
                lset, test_set, batch_size,
                num_epochs, convex_epochs, test_on_train=test_on_train)
            cls2.model = cls2.best_model
            cls2.test(test_set, 'Test')

        if cls_loss_drop:
            labeled_set.drop_and(cls, fraction=1)
        if true_noise_drop:
            labeled_set.drop_noise()
        labeled_set.weight_tensor[:] *= 1/2
        if uncertainty:
            UncertaintyQuery().query(
                unlabeled_set, labeled_set, query_batch_size,
                cls, incr_pool_size, weight_ratio)
        elif compare_with_perfect:
            DisagreementQuery().query(
                unlabeled_set, labeled_set,
                query_batch_size, [perfect_cls, cls], weight_ratio)
        cls.model = cls.best_model
        cls.test(test_set, 'Test')
        if not uncertainty and not compare_with_perfect:
            ClsDisagreementQuery().query(
                unlabeled_set, labeled_set, query_batch_size,
                cls, weight_ratio)

    print('\nRandom Query'.format(incr))
    labeled_set_rand.is_used_tensor[:] = 1
    if incr < 0:
        labeled_set_rand.drop_noise()
    if local_noise_drop:
        labeled_set_rand.drop_local_inconsitent()
    cls_rand.train(
        labeled_set_rand, test_set, batch_size,
        num_epochs, convex_epochs, test_on_train=test_on_train)
    cls_rand.model = cls_rand.best_model
    cls_rand.test(test_set, 'Test')

    if incr == 0:
        cls_rand2 = deepcopy(cls_rand)
    else:
        lsetr = deepcopy(labeled_set_rand)
        lsetr.weight_tensor[:] = 1
        print('')
        cls_rand2.train(
            lsetr, test_set, batch_size,
            num_epochs, convex_epochs, test_on_train=test_on_train)
        cls_rand2.model = cls_rand2.best_model
        cls_rand2.test(test_set, 'Test')

    if cls_loss_drop:
        labeled_set_rand.drop_and(cls_rand, fraction=1)
    if true_noise_drop:
        labeled_set_rand.drop_noise()
    # if incr < incr_times:
        # print('drop all')
        # labeled_set_rand.is_used_tensor[:] = 0
    # else:
        # labeled_set_rand.is_used_tensor[:] = 1
        # labeled_set_rand.drop_noise()
    labeled_set_rand.weight_tensor[:] *= 1/2
    RandomQuery().query(
        unlabeled_set_rand, labeled_set_rand, query_batch_size, init_weight)


print('\nincr {}'.format(incr_times))
convex_epochs = (init_convex_epochs
                 if incr_times == 0
                 else retrain_convex_epochs)
num_epochs = init_epochs if incr_times == 0 else retrain_epochs

if not args.no_active:
    print('\nActive Query'.format(incr))
    if local_noise_drop:
        labeled_set.is_used_tensor[:] = 1
        labeled_set.drop_local_inconsitent()
    cls.train(labeled_set, test_set, batch_size, num_epochs,
              convex_epochs, test_on_train=test_on_train)
    cls.model = cls.best_model
    cls.test(test_set, 'Test')

    lset = deepcopy(labeled_set)
    lset.weight_tensor[:] = 1
    print('')
    cls2.train(
        lset, test_set, batch_size,
        num_epochs, convex_epochs, test_on_train=test_on_train)
    cls2.model = cls2.best_model
    cls2.test(test_set, 'Test')

print('\nRandom Query'.format(incr_times))
if local_noise_drop:
    labeled_set_rand.is_used_tensor[:] = 1
    labeled_set_rand.drop_local_inconsitent()
cls_rand.train(
    labeled_set_rand, test_set, batch_size,
    num_epochs, convex_epochs, test_on_train=test_on_train)
cls_rand.model = cls_rand.best_model
cls_rand.test(test_set, 'Test')

lsetr = deepcopy(labeled_set_rand)
lsetr.weight_tensor[:] = 1
print('')
cls_rand2.train(
    lsetr, test_set, batch_size,
    num_epochs, convex_epochs, test_on_train=test_on_train)
cls_rand2.model = cls_rand2.best_model
cls_rand2.test(test_set, 'Test')


if incr_times > 0:

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
        cls.model = cls.best_model
        cls.test(test_set, 'Test')

    print('\nRandomly Selected Points')
    # labeled_set_rand.is_used_tensor[:] = 1
    # labeled_set_rand.drop_local_inconsitent()
    labeled_set_rand.weight_tensor[:] = 1
    cls_rand.train(
        labeled_set_rand, test_set, batch_size,
        init_epochs, init_convex_epochs,
        test_on_train=test_on_train)
    cls_rand.model = cls_rand.best_model
    cls_rand.test(test_set, 'Test')
