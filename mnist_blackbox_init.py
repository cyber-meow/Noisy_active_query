import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import argparse
# import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
# from sklearn.decomposition import PCA

import settings
from classifier import Classifier
from dataset import WeightedTensorDataset


init_weight = 30
init_p_size = 750  # 650
init_n_size = 750  # 450
init_p_un_size = 0  # 1000
init_n_un_size = 0  # 500
uncertainty_pool = 3500

pho_p = 0.5
pho_n = 0

batch_size = 200
num_clss = 2
learning_rate = 5e-4
incr_times = 2
test_on_train = False

retrain_epochs = 80
convex_epochs = 6
query_batch_size = 100
reduced_sample_size = 5
used_size = 450

n_pca_components = 784


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

# pca = PCA(n_components=n_pca_components)
# train_data = pca.fit_transform(train_data.reshape(-1, 784))

train_data = train_data[used_idxs]
train_labels = train_labels[used_idxs]

train_data = torch.from_numpy(train_data).unsqueeze(1).float()
train_labels = torch.from_numpy(train_labels).unsqueeze(1).float()
training_set = WeightedTensorDataset(
    train_data, train_labels, init_weight * torch.ones(len(train_data), 1))


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


model = Net().cuda() if args.cuda else Net()
cls = Classifier(model)
cls.train(training_set, test_set, batch_size, 2, 1)
output = cls.model(Variable(train_data).type(settings.dtype)).cpu()
probs = F.sigmoid(output).data.numpy().reshape(-1)
sorted_margin = torch.from_numpy(np.argsort(np.abs(probs-0.5)))

# margin_sorted_data = train_data[sorted_margin]

# grid_img = torchvision.utils.make_grid(
#     torch.cat((margin_sorted_data[:32], margin_sorted_data[-32:]), 0))
# plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))
# plt.show()


tmp_labels = train_labels.numpy().reshape(-1)
p_idxs = tmp_labels == 1
n_idxs = tmp_labels == -1

un_idxs = np.zeros(len(train_data))
un_idxs[sorted_margin[:uncertainty_pool]] = True

p_un = np.argwhere(np.logical_and(p_idxs, un_idxs)).reshape(-1)
drawn = np.random.choice(p_un, init_p_un_size, replace=False)

n_un = np.argwhere(np.logical_and(n_idxs, un_idxs)).reshape(-1)
drawn2 = np.random.choice(n_un, init_n_un_size, replace=False)

# plt_data = train_data[torch.from_numpy(
#     np.concatenate([drawn[:32], drawn2[:32]]))]
# print(plt_data)
# grid_img = torchvision.utils.make_grid(plt_data)
# plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))
# plt.show()

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
    given_data, given_labels, init_weight * torch.ones(len(given_data), 1))
print(len(labeled_set))


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        self.linear = nn.Linear(n_pca_components, 1)

    def forward(self, x):
        y_pred = self.linear(x.view(-1, n_pca_components))
        return y_pred


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
