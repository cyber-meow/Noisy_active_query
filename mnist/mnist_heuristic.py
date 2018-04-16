import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import argparse
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.decomposition import PCA
from collections import OrderedDict
from scipy.spatial import distance

import dataset
import settings
from classifier import Classifier
from mnist.basics import Net, Linear


pho_p = 0.4
pho_n = 0.1

batch_size = 40
learning_rate = 1e-3
weight_decay = 1e-2

convex_epochs = 5
retrain_epochs = 100
test_on_train = True

num_clss = 1
init_size = 500

used_size = 200
incr_times = 0
query_batch_size = 40
reduced_sample_size = 4

init_weight = 1
weight_ratio = 2

use_CNN = True
kcenter = False

n_pca_components = 784

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

data_init = (dataset.datasets_initialization_kcenter
             if kcenter
             else dataset.datasets_initialization)

unlabeled_set, labeled_set = data_init(
    train_data, train_labels, init_size, init_weight, pho_p, pho_n)
train_labels = torch.sign(labeled_set.target_tensor).numpy()
train_labels_clean = labeled_set.label_tensor.numpy()


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


def confidence_scores(data, votes):
    p_d = distance.squareform(distance.pdist(data))
    sigma = np.mean(np.sort(p_d, axis=1)[:, :5])
    K = np.exp(-p_d**2/sigma**2)
    votes = votes.reshape(-1)
    score = np.sum(K * votes, axis=1) * votes
    score = 2*score/np.std(score)
    conf = 1/(1+np.exp(-score))
    # a = (pho_p - pho_n)/2
    # b = (pho_p + pho_n)/2
    # pho_y = a * np.sign(votes) + b
    # pho_ny = pho_p + pho_n - pho_y
    # class_conf = (1-pho_y)/(1-pho_y+pho_ny)
    # return 1 - np.sqrt((1-conf)*(1-class_conf))
    return conf


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


cls = create_new_classifier()
cls.train(labeled_set, test_set, batch_size,
          retrain_epochs, convex_epochs,
          test_on_train=test_on_train)
out = cls.model(Variable(labeled_set.data_tensor).type(settings.dtype))
sigmoid = nn.Sigmoid()
cls_conf = sigmoid(out*Variable(
            torch.sign(labeled_set.target_tensor)
            ).type(settings.dtype)).data.numpy().reshape(-1)


conf = confidence_scores(
    labeled_set.data_tensor.numpy().reshape(-1, 784),
    labeled_set.target_tensor.numpy())
print(np.sum(conf < 0.5))

# grid_img = torchvision.utils.make_grid(
#     labeled_set.data_tensor[torch.from_numpy(np.argsort(conf)[:10])])
# plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))

diff = (train_labels != train_labels_clean).reshape(-1)
plt.plot(diff[np.argsort(conf)], label='conf hit')
plt.plot(diff[np.argsort(cls_conf)], '--', label='cls conf hit', alpha=0.6)
plt.plot(np.sort(conf), label='conf')
plt.plot(np.sort(cls_conf), label='cls conf')
plt.legend()

plt.figure()
plt.plot(cls.train_accuracies, label='train accuracy')
plt.plot(cls.test_accuracies, label='test accuracy')
# plt.plot(cls.high_loss_fractions, label='fraction of high loss samples')
plt.plot([100-conf for conf in cls.critic_confs], label='critic conf')
plt.plot([100-conf for conf in cls.confs], label='conf')
# plt.plot(cls.critic_losses, label='critic logistic loss')
plt.plot(cls.high_loss_errors, label='high loss error')
plt.legend()
plt.show()
