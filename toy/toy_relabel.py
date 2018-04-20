import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from copy import deepcopy

import dataset
from toy.basics import Net, ToyClassifier
from active_query import DisagreementQuery, greatest_impact
from active_query import HeuristicRelabel, ClsHeuristicRelabel

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

num_clss = 3
init_size = 80
kcenter = False

init_weight = 1
weight_ratio = 2

query_times = 5
relabel_size = 0
random_flipped_size = 5
incr_size = 10
cls_uncertainty = True


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
y_clean = labeled_set.label_tensor.numpy().reshape(-1)

cx, cy = np.array(x_init).T
plt.scatter(cx, cy, s=20, color='yellow')
cx, cy = x_init[y_init == -1].T
plt.scatter(cx, cy, s=40, c='black', alpha=0.2, marker='+')
plt.pause(0.05)


cls = create_new_classifier()
# IWALQuery = IWALQuery()

conts = []
cm = plt.get_cmap('gist_rainbow')
cm2 = plt.get_cmap('winter')


for query in range(query_times+1):

    print('\nQuery {}'.format(query))

    if query >= 1:
        for cont in conts:
            for coll in cont.collections:
                coll.remove()
        conts = []

    cls.train(labeled_set, test_set, retrain_epochs,
              convex_epochs, test_interval=10,
              print_interval=3000, test_on_train=True)
    conts.append(cls.model.plot_boundary(ax, colors=[cm(0)]))
    plt.pause(0.05)

    labeled_set.is_used_tensor[:] = 1

    if cls_uncertainty:
        flipped_idxs_sets, drop_idxs = ClsHeuristicRelabel().diverse_flipped(
            labeled_set, num_clss-1, random_flipped_size, cls, pho_p, pho_n)
    else:
        flipped_idxs_sets, drop_idxs = HeuristicRelabel().diverse_flipped(
            labeled_set, num_clss-1, random_flipped_size, 5, pho_p, pho_n)
    print(drop_idxs)
    # labeled_set.is_used_tensor[:] = 1

    x_dropped = labeled_set.data_tensor.numpy()[drop_idxs]
    dx, dy = x_dropped.T
    plt.scatter(dx, dy, s=40, marker='x', label='{} dropped'.format(query))
    # plt.legend()
    plt.pause(0.05)

    idxs_clss = []
    clss = [cls]
    pss = []
    x = labeled_set.data_tensor.numpy()

    for i, (idxs, flipped_set) in enumerate(flipped_idxs_sets):
        ix, iy = x[idxs].T
        pss.append(
            plt.scatter(ix, iy, s=20, alpha=0.8, marker='d',
                        color=cm((i+1)/num_clss)))
        plt.pause(0.05)
        new_cls = deepcopy(cls)
        new_cls.train(flipped_set, test_set, retrain_epochs,
                      convex_epochs, test_interval=10,
                      print_interval=3000, test_on_train=True)
        conts.append(
            new_cls.model.plot_boundary(ax, colors=[cm((i+1)/num_clss)]))
        plt.pause(0.05)
        idxs_clss.append((idxs, new_cls))
        clss.append(new_cls)

    if relabel_size != 0:
        relabel_idxs = greatest_impact(cls, idxs_clss, unlabeled_set)
        print(relabel_idxs)
        labeled_set.query(relabel_idxs)

    y = torch.sign(labeled_set.target_tensor).numpy().reshape(-1)

    if incr_size != 0:
        x_selected, y_selected, _ = DisagreementQuery().query(
            unlabeled_set, labeled_set, incr_size, clss, weight_ratio)
        x_selected = x_selected.numpy()
        y_selected = y_selected.numpy().reshape(-1)
        if relabel_size != 0:
            x_selected = np.concatenate([x_selected, x[relabel_idxs]])
            # print(labeled_set.target_tensor.numpy().reshape(-1)[relabel_idxs])
            y_selected = np.concatenate([y_selected, y[relabel_idxs]])
    else:
        assert(relabel_size != 0)
        x_selected = x[relabel_idxs]
        y_selected = y[relabel_idxs]

    sx, sy = x_selected.T
    plt.scatter(sx, sy, s=10, alpha=0.6, color=cm2(query/query_times),
                label='{} queried'.format(query))
    sx, sy = x_selected[y_selected == -1].T
    print(y_selected)
    plt.scatter(sx, sy, s=50, alpha=0.7, c='black', marker='+')
    # plt.legend()
    plt.pause(0.05)

    if relabel_size != 0:
        removed_data, remove_labels = labeled_set.remove_no_effect()
        if removed_data is not None:
            unlabeled_set.data_tensor = torch.cat(
                (unlabeled_set.data_tensor, removed_data), 0)
            unlabeled_set.target_tensor = torch.cat(
                (unlabeled_set.target_tensor, remove_labels), 0)

    while not plt.waitforbuttonpress(1):
        pass

    for ps in pss:
        ps.remove()
    plt.pause(0.05)

plt.figure()
plt.plot(cls.train_accuracies, label='train accuracy')
plt.plot(cls.test_accuracies, label='test accuracy')
# plt.plot(cls.high_loss_fractions, label='fraction of high loss samples')
plt.plot(cls.critic_confs, label='critic conf')
plt.legend()

while not plt.waitforbuttonpress(1):
    pass
