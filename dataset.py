import torch
import torch.utils.data
import numpy as np
from copy import deepcopy
from k_center import KCenter


class WeightedTensorDataset(torch.utils.data.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        weight_tensor (Tensor): contains sample weights for loss evaluation.
    """

    def __init__(self, data_tensor, target_tensor, weight_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        assert data_tensor.size(0) == weight_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.weight_tensor = weight_tensor

    def __getitem__(self, index):
        return (self.data_tensor[index],
                self.target_tensor[index],
                self.weight_tensor[index])

    def __len__(self):
        return self.data_tensor.size(0)

    def update(self, data_tensor, target_tensor, weight_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        assert data_tensor.size(0) == weight_tensor.size(0)
        self.data_tensor = torch.cat(
            (self.data_tensor, data_tensor), 0)
        self.target_tensor = torch.cat(
            (self.target_tensor, target_tensor), 0)
        self.weight_tensor = torch.cat(
            (self.weight_tensor, weight_tensor), 0)


def label_corruption(labels, pho_p=0, pho_n=0):
    corrupted_labels = deepcopy(labels)
    for i, label in enumerate(labels):
        assert label[0] == 1 or label[0] == -1
        if label[0] == 1 and np.random.random() < pho_p:
            # print('flip +1')
            corrupted_labels[i] = -1
        elif np.random.random() < pho_n:
            # print('flip -1')
            corrupted_labels[i] = 1
    return corrupted_labels


def datasets_initialization(data, labels, init_size, init_weight):
    if not torch.is_tensor(data):
        data = torch.from_numpy(data).float()
    if not torch.is_tensor(labels):
        labels = torch.from_numpy(labels).float()
    idxs = torch.from_numpy(np.random.permutation(data.size(0)))
    data = data[idxs]
    labels = labels[idxs]
    unlabeled_set = torch.utils.data.TensorDataset(
        data[init_size:], labels[init_size:])
    labeled_set = WeightedTensorDataset(
        data[:init_size], labels[:init_size],
        init_weight * torch.ones(init_size, 1))
    return unlabeled_set, labeled_set


def datasets_initialization_kcenter(
        data, labels, init_size, init_weight, pho_p=0, pho_n=0):
    if torch.is_tensor(data):
        data = data.numpy()
    if torch.is_tensor(labels):
        labels = labels.numpy()
    kcenter = KCenter(data, labels)
    for i in range(init_size-1):
        # if i % 10 == 0:
        #     print(i)
        kcenter.select_one()
    unlabeled_set = torch.utils.data.TensorDataset(
        torch.from_numpy(kcenter.pool).float(),
        torch.from_numpy(kcenter.pool_y).float())
    labeled_set = WeightedTensorDataset(
        torch.from_numpy(kcenter.selected).float(),
        torch.from_numpy(kcenter.selected_y).float(),
        init_weight * torch.ones(init_size, 1))
    return unlabeled_set, labeled_set
