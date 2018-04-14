import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance

import dataset
from toy.basics import Net, ToyClassifier


moons = True
n_positive = 10000
n_negative = 10000
n = n_positive + n_negative

pho_p = 0.5
pho_n = 0
pho_p_c = pho_p
pho_n_c = pho_n

learning_rate = 5e-3
weight_decay = 1e-3

convex_epochs = 500
retrain_epochs = 12000

init_weight = 1
init_size = 90
kcenter = False

corr_times = 1
corr_size = 10

load = False
save = False


def create_new_classifier():
    model = Net()
    cls = ToyClassifier(
            model,
            pho_p=pho_p_c,
            pho_n=pho_n_c,
            lr=learning_rate,
            weight_decay=weight_decay)
    return cls


if os.path.exists('datasets/toy/train_data.npy') and load:
    x_all = np.load('datasets/toy/train_data.npy')
    y_all = np.load('datasets/toy/train_labels_clean.npy')
    y_all_corrupted = np.load('datasets/toy/train_labels.npy')

else:
    if moons:
        x_all, y_all = datasets.make_moons(n, noise=0.07)
    else:
        x_all, y_all = datasets.make_circles(n, noise=0.03)
    y_all = (y_all*2-1).reshape(-1, 1)

    y_all_corrupted = dataset.label_corruption(y_all, pho_p, pho_n)
    y = np.concatenate([y_all_corrupted, y_all], axis=1)

    if save:
        np.save('datasets/toy/train_data', x_all)
        np.sqve('datasets/toy/train_labels_clean.npy', y_all)
        np.save('datasets/toy/train_labels', y_all_corrupted)

if kcenter:
    unlabeled_set, labeled_set = dataset.datasets_initialization_kcenter(
        x_all, y, init_size, init_weight)
else:
    unlabeled_set, labeled_set = dataset.datasets_initialization(
        x_all, y, init_size, init_weight)


if os.path.exists('datasets/toy/test_data.npy') and load:
    x_test = np.load('datasets/toy/test_data.npy')
    y_test = np.load('datasets/toy/test_labels.npy')
else:
    if moons:
        x_test, y_test = datasets.make_moons(n, noise=0.07)
    else:
        x_test, y_test = datasets.make_circles(n, noise=0.03)
    y_test = (y_test*2-1).reshape(-1, 1)
    if save:
        np.save('datasets/toy/test_data', x_test)
        np.save('datasets/toy/test_labels', y_test)

test_set = torch.utils.data.TensorDataset(
    torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())


fig, ax = plt.subplots()

plt.ion()
plt.show()

negative_samples = x_all[y_all.reshape(-1) == 1]
positive_samples = x_all[y_all.reshape(-1) == -1]

px, py = np.array(positive_samples).T
nx, ny = np.array(negative_samples).T
plt.scatter(px, py, color='mistyrose', s=3)
plt.scatter(nx, ny, color='turquoise', s=3)
plt.pause(0.05)

x_init = labeled_set.data_tensor.numpy()
y_init = labeled_set.target_tensor.numpy()
labeled_set.target_tensor = torch.from_numpy(y_init[:, 0].reshape(-1, 1))

cx, cy = np.array(x_init).T
plt.scatter(cx, cy, s=10, color='yellow')
cx, cy = x_init[y_init[:, 0] == -1].T
plt.scatter(cx, cy, s=40, c='black', alpha=0.2, marker='+')
plt.pause(0.05)


cls = create_new_classifier()
cm = plt.get_cmap('gist_rainbow')

cont = None


def give_correct(y, k):
    s = y[:, 0] + y[:, 1]
    diff_indices = np.argwhere(s == 0).reshape(-1)
    k = min(k, len(diff_indices))
    if k != 0:
        flipped_indices = np.random.choice(diff_indices, k, replace=False)
        y[flipped_indices, 0] = y[flipped_indices, 1]
        return flipped_indices
    return None


def confidence_scores(data, labels):
    p_d = distance.squareform(distance.pdist(data))
    sigma = 2*np.mean(np.sort(p_d, axis=1)[:, :5])
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
    return conf


def density_estimation(labeled, unlabeled):
    p_d = distance.squareform(distance.pdist(labeled))
    sigmas = np.mean(np.sort(p_d, axis=1)[:, 1:4], axis=1)
    print(sigmas)
    c_d = distance.cdist(labeled, unlabeled)
    # labeled_K = np.exp(-p_d**2/sigma**2)
    unlabeled_K = np.exp(-c_d**2/sigmas[:, None]**2)
    # labeled_density = np.sum(labeled_K, axis=1)
    unlabeled_impact = np.sum(unlabeled_K, axis=1)
    return unlabeled_impact/np.mean(unlabeled_impact)
    # return np.log(1+np.exp(-density_ratio))


for corr in range(corr_times):

    print('\ncorrection {}'.format(corr))

    cls.train(labeled_set, test_set, retrain_epochs,
              convex_epochs, test_interval=10, test_on_train=True)
    if corr >= 1:
        cont.collections[0].set_linewidth(1)
        cont.collections[0].set_alpha(0.3)
    cont = cls.model.plot_boundary(ax, colors=[cm(corr/corr_times)])
    plt.pause(0.05)

    conf = confidence_scores(
        labeled_set.data_tensor.numpy(), labeled_set.target_tensor.numpy())
    dr = density_estimation(labeled_set.data_tensor.numpy(), x_all)
    x = labeled_set.data_tensor.numpy()

    # diff = y_init[:, 0] != y_init[:, 1]
    # plt.plot(diff[np.argsort(conf)])

    sizes = (1-conf) * dr * 30
    sizes[sizes < np.percentile(sizes, 75)] = 0
    print(sorted(sizes))
    lx, ly = x.T
    plt.scatter(lx, ly, s=sizes, alpha=0.4, label='{} conf'.format(corr))
    plt.legend()
    plt.pause(0.05)
    while not plt.waitforbuttonpress(1):
        pass

    sizes = (1-conf) * 60
    sizes[sizes < np.percentile(sizes, 75)] = 0
    print(sorted(sizes))
    lx, ly = x.T
    plt.scatter(lx, ly, s=sizes, color='red', marker='x',
                alpha=0.2, label='{} conf'.format(corr))
    plt.legend()
    plt.pause(0.05)
    while not plt.waitforbuttonpress(1):
        pass

    '''
    sizes = dr * 60
    sizes[sizes < np.percentile(sizes, 70)] = 0
    print(sorted(sizes))
    lx, ly = x.T
    plt.scatter(lx, ly, s=sizes, color='green', marker='o',
                alpha=0.4, label='{} conf'.format(corr))
    plt.legend()
    plt.pause(0.05)
    '''

    flipped_indices = give_correct(y_init, corr_size)

    if flipped_indices is None:
        break

    labeled_set.target_tensor = torch.from_numpy(y_init[:, 0].reshape(-1, 1))
    x_selected = x_init[flipped_indices]
    labeled_set.weight_tensor[torch.from_numpy(flipped_indices)] = 0

    sx, sy = x_selected.T
    plt.scatter(sx, sy, s=5, alpha=0.5, label='{}'.format(corr))
    plt.legend()
    plt.pause(0.05)

    while not plt.waitforbuttonpress(1):
        pass

plt.figure()
plt.plot(cls.train_accuracies, label='train accuracy')
plt.plot(cls.test_accuracies, label='test accuracy')
plt.plot(cls.high_loss_fractions, label='fraction of high loss samples')
plt.plot(cls.critic_confs, label='critic conf')
plt.legend()

while not plt.waitforbuttonpress(1):
    pass
