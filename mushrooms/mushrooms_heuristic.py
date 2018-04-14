import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from scipy.spatial import distance

import dataset
import settings
from classifier import Classifier


pho_p = 0.1
pho_n = 0.1

batch_size = 50
learning_rate = 5e-3
weight_decay = 1e-2

convex_epochs = 10
retrain_epochs = 300
test_on_train = True

num_clss = 5
init_size = 100

used_size = 75
incr_times = 0
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
train_labels_clean = torch.from_numpy(y_train.values).unsqueeze(1).float()
train_labels_corrupted = dataset.label_corruption(
    train_labels_clean, pho_p, pho_n)
train_labels = torch.cat(
    [train_labels_corrupted, train_labels_clean], dim=1)

data_init = (dataset.datasets_initialization_kcenter
             if kcenter
             else dataset.datasets_initialization)

unlabeled_set, labeled_set = data_init(
    train_data, train_labels, init_size, init_weight)
train_labels = labeled_set.target_tensor.numpy()
labeled_set.target_tensor = torch.from_numpy(train_labels[:, 0, None])


test_data = torch.from_numpy(X_test.values).float()
test_set = torch.utils.data.TensorDataset(
    test_data, torch.from_numpy(y_test.values).unsqueeze(1).float())


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Linear(117, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def confidence_scores(data, labels):
    p_d = distance.squareform(distance.pdist(data))
    sigma = np.mean(np.sort(p_d, axis=1)[:, :5])
    K = np.exp(-p_d**2/sigma**2)
    labels = labels.reshape(-1)
    a = (pho_p - pho_n)/2
    b = (pho_p + pho_n)/2
    pho_y = a * labels + b
    pho_ny = pho_p + pho_n - pho_y
    weights = 1 - pho_ny + pho_y
    score = np.sum(K * weights * labels, axis=1) * labels
    score = 2*score/np.std(score)
    conf = 1/(1+np.exp(-score))
    # class_conf = (1-pho_y)/(1-pho_y+pho_ny)
    return conf  # * class_conf


def create_new_classifier():
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
cls_conf = sigmoid(
            out*Variable(labeled_set.target_tensor).type(settings.dtype)
            ).data.numpy().reshape(-1)


conf = confidence_scores(
    labeled_set.data_tensor.numpy(),
    train_labels[:, 0, None])
print(train_labels[:, 0][conf < 0.5])
print(np.sum(conf < 0.5))

# grid_img = torchvision.utils.make_grid(
#     labeled_set.data_tensor[torch.from_numpy(np.argsort(conf)[:10])])
# plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))

diff = (train_labels[:, 0] != train_labels[:, 1]).reshape(-1)
plt.plot(diff[np.argsort(conf)], label='conf hit')
plt.plot(diff[np.argsort(cls_conf)], '--', label='cls conf hit', alpha=0.6)
plt.plot(np.sort(conf), label='conf')
plt.plot(np.sort(cls_conf), label='cls conf')
plt.legend()

plt.figure()
plt.plot(cls.train_accuracies, label='train accuracy')
plt.plot(cls.test_accuracies, label='test accuracy')
plt.plot(cls.high_loss_fractions, label='fraction of high loss samples')
plt.plot(cls.critic_losses, label='critic loss')
plt.legend()
plt.show()
