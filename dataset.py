import torch
import numpy as np


class WeigthedTensorDataset(torch.utils.data.Dataset):
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


def datasets_initialization(
        data, labels, init_size, init_weight, pho_p=0, pho_n=0):
    idxs = torch.from_numpy(np.random.permutation(data.size(0)))
    data = data[idxs]
    labels = labels[idxs]
    for i, label in enumerate(labels):
        assert label[0] == 1 or label[0] == -1
        if label[0] == 1 and np.random.random() < pho_p:
            # print('flip +1')
            labels[i] = -1
        elif np.random.random() < pho_n:
            # print('flip -1')
            labels[i] = 1
    unlabeled_set = torch.utils.data.TensorDataset(
        data[init_size:], labels[init_size:])
    labeled_set = WeigthedTensorDataset(
        data[:init_size], labels[:init_size],
        init_weight * torch.ones(init_size, 1))
    return unlabeled_set, labeled_set
