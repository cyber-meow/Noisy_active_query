import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

import dataset
from active_query import UncertaintyQuery, DisagreementQuery
from active_query import HeuristicRelabel
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

convex_epochs = 6000
retrain_epochs = 12000

init_weight = 1
init_size = 80
kcenter = False
uncertainty = False

query_batch_size = 5
incr_times = 10
incr_pool_size = 1500


def create_new_classifier():
    model = Net()
    cls = ToyClassifier(
            model,
            pho_p=pho_p_c,
            pho_n=pho_n_c,
            lr=learning_rate,
            weight_decay=weight_decay)
    return cls


if moons:
    x_all, y_all = datasets.make_moons(n, noise=0.07)
else:
    x_all, y_all = datasets.make_circles(n, noise=0.03)
y_all = (y_all*2-1).reshape(-1, 1)

if kcenter:
    unlabeled_set, labeled_set = dataset.datasets_initialization_kcenter(
        x_all, y_all, init_size, init_weight, pho_p, pho_n)
else:
    unlabeled_set, labeled_set = dataset.datasets_initialization(
        x_all, y_all, init_size, init_weight, pho_p, pho_n)

training_set = torch.utils.data.TensorDataset(
    torch.from_numpy(x_all).float(), torch.from_numpy(y_all).float())

if moons:
    x_test, y_test = datasets.make_moons(n, noise=0.07)
else:
    x_test, y_test = datasets.make_circles(n, noise=0.03)
y_test = (y_test*2-1).reshape(-1, 1)

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
y_init = torch.sign(labeled_set.target_tensor).numpy().reshape(-1)

cx, cy = np.array(x_init).T
plt.scatter(cx, cy, s=3, color='yellow')
cx, cy = x_init[y_init == -1].T
plt.scatter(cx, cy, s=3, c='black', alpha=0.2)
plt.pause(0.05)


if not uncertainty:
    perfect_cls = create_new_classifier()
    perfect_cls.train(training_set, test_set, 2000, 2000, test_interval=400)
    perfect_cls.model.plot_boundary(ax, colors=['gold'])
    plt.pause(0.05)


cls = create_new_classifier()
cm = plt.get_cmap('gist_rainbow')

cont = None


for incr in range(incr_times+1):

    print('\nincr {}'.format(incr))

    cls.train(labeled_set, test_set, retrain_epochs, convex_epochs,
              test_interval=10, print_interval=3000, test_on_train=True)
    if incr >= 1:
        cont.collections[0].set_linewidth(1)
        cont.collections[0].set_alpha(0.3)
    cont = cls.model.plot_boundary(ax, colors=[cm(incr/incr_times)])
    plt.pause(0.05)

    labeled_set.is_used_tensor[:] = 1
    print(torch.sum(labeled_set.is_used_tensor))
    flipped_idxs_sets, drop_idxs = HeuristicRelabel().diverse_flipped(
        labeled_set, 0, 0, 5, pho_p, pho_n)
    print(drop_idxs)
    print(torch.sum(labeled_set.is_used_tensor))

    x_dropped = labeled_set.data_tensor.numpy()[drop_idxs]
    dx, dy = x_dropped.T
    plt.scatter(dx, dy, s=40, marker='x', label='{} dropped'.format(incr))
    plt.pause(0.05)

    if uncertainty:
        cls.model = cls.best_model
        x_selected, y_selected, _ = UncertaintyQuery().query(
            unlabeled_set, labeled_set, query_batch_size, cls,
            incr_pool_size, init_weight)
    else:
        x_selected, y_selected, _ = DisagreementQuery().query(
            unlabeled_set, labeled_set, query_batch_size,
            [perfect_cls, cls], init_weight)
        cls.model = cls.best_model

    cls.test(test_set, 'Test')
    x_selected = x_selected.numpy()
    y_selected = y_selected.numpy().reshape(-1)
    sx, sy = x_selected.T
    plt.scatter(sx, sy, s=7, label='{}'.format(incr))
    sx, sy = x_selected[y_selected == -1].T
    plt.scatter(sx, sy, s=25, c='black', alpha=0.2)
    # plt.legend()
    plt.pause(0.05)


while not plt.waitforbuttonpress(1):
    pass
