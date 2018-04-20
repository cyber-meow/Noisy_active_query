import torch
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from scipy.spatial import distance
from copy import deepcopy

import settings
from k_center import KCenter


class WeightedTensorDataset(torch.utils.data.Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        label_tensor (Tensor): contains true labels.
        weight_tensor (Tensor): contains sample weights for loss evaluation.
    """

    def __init__(self, data_tensor, label_tensor, weight_tensor,
                 pho_p=0, pho_n=0):
        assert data_tensor.size(0) == label_tensor.size(0)
        assert data_tensor.size(0) == weight_tensor.size(0)
        self.pho_p = pho_p
        self.pho_n = pho_n
        self.p_weight = 1 - pho_n + pho_p
        self.n_weight = 1 - pho_p + pho_n
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.weight_tensor = weight_tensor
        self.target_tensor = torch.zeros_like(weight_tensor)
        self.is_used_tensor = torch.ones_like(weight_tensor)
        self.query(np.arange(data_tensor.size(0)))

    def __getitem__(self, index):
        return (self.data_tensor[index],
                self.weight_tensor[index] * self.target_tensor[index]
                * self.is_used_tensor[index])

    def __len__(self):
        return self.data_tensor.size(0)

    def update(self, data_tensor, label_tensor, weight_tensor):
        assert data_tensor.size(0) == label_tensor.size(0)
        assert data_tensor.size(0) == weight_tensor.size(0)
        self.data_tensor = torch.cat(
            (self.data_tensor, data_tensor), 0)
        self.label_tensor = torch.cat(
            (self.label_tensor, label_tensor), 0)
        self.weight_tensor = torch.cat(
            (self.weight_tensor, weight_tensor), 0)
        old_length = self.target_tensor.size(0)
        self.target_tensor = torch.cat(
            (self.target_tensor, torch.zeros_like(weight_tensor)), 0)
        self.is_used_tensor = torch.cat(
            (self.is_used_tensor, torch.ones_like(weight_tensor)), 0)
        self.query(np.arange(old_length, self.target_tensor.size(0)))
        update_labels = torch.sign(self.target_tensor[torch.LongTensor(
            np.arange(old_length, self.target_tensor.size(0)))])
        return data_tensor, update_labels

    def modify(self, indices):
        new_set = deepcopy(self)
        for idx in indices:
            assert(self.target_tensor[idx][0] != 0)
            if self.target_tensor[idx][0] > 0:
                new_set.target_tensor[idx] -= self.n_weight
            else:
                new_set.target_tensor[idx] += self.p_weight
        return new_set

    def query(self, indices):
        for idx in indices:
            if self.label_tensor[idx][0] == 1:
                if np.random.random() < self.pho_p:
                    self.target_tensor[idx] -= self.n_weight
                else:
                    self.target_tensor[idx] += self.p_weight
            else:
                if np.random.random() < self.pho_n:
                    self.target_tensor[idx] += self.p_weight
                else:
                    self.target_tensor[idx] -= self.n_weight
        # print(self.target_tensor)

    def drop(self, indices):
        if len(indices) == 0:
            return
        if not torch.is_tensor(indices):
            indices = torch.from_numpy(indices)
        self.is_used_tensor[indices] = 0

    def drop_large_loss(self, cls, fraction=1/3):
        x = Variable(self.data_tensor).type(settings.dtype)
        target = Variable(
            torch.sign(self.target_tensor)).type(settings.dtype)
        output = cls.model(x)
        losses = cls.basic_loss(
            output*target, False).data.cpu().numpy().reshape(-1)
        noise_rate = (self.pho_p + self.pho_n)/2 * 100
        th = np.percentile(losses, 100-noise_rate*fraction)
        drop_indices = np.argwhere(losses > th).reshape(-1)
        self.drop(drop_indices)
        drop_indices_t = torch.from_numpy(drop_indices)
        noise = (torch.sign(self.target_tensor)[drop_indices_t]
                 != self.label_tensor[drop_indices_t])
        print('noise/drop: {}/{}'.format(torch.sum(noise), len(drop_indices)))

    def drop_local_inconsitent(self, k=5, fraction=1/3):
        p_d = distance.squareform(distance.pdist(
            self.data_tensor.numpy().reshape(len(self), -1)))
        sigma = np.mean(np.sort(p_d, axis=1)[:, :k])
        K = np.exp(-p_d**2/sigma**2)
        votes = self.target_tensor.numpy().reshape(-1)
        score = np.sum(K * votes, axis=1) * votes
        score = 2*score/np.std(score)
        conf = 1/(1+np.exp(-score))
        noise_rate = (self.pho_p + self.pho_n)/2 * 100
        th = np.percentile(conf, noise_rate*fraction)
        drop_indices = np.argwhere(conf < th).reshape(-1)
        self.drop(drop_indices)
        drop_indices_t = torch.from_numpy(drop_indices)
        noise = (torch.sign(self.target_tensor)[drop_indices_t]
                 != self.label_tensor[drop_indices_t])
        print('noise/drop: {}/{}'.format(torch.sum(noise), len(drop_indices)))

    def drop_and(self, cls, k=5, fraction=1/3):

        x = Variable(self.data_tensor).type(settings.dtype)
        target = Variable(
            torch.sign(self.target_tensor)).type(settings.dtype)
        output = cls.model(x)
        losses = cls.basic_loss(
            output*target, False).data.cpu().numpy().reshape(-1)

        p_d = distance.squareform(distance.pdist(
            self.data_tensor.numpy().reshape(len(self), -1)))
        sigma = np.mean(np.sort(p_d, axis=1)[:, :k])
        K = np.exp(-p_d**2/sigma**2)
        votes = self.target_tensor.numpy().reshape(-1)
        score = np.sum(K * votes, axis=1) * votes
        score = 2*score/np.std(score)
        conf = 1/(1+np.exp(-score))

        noise_rate = (self.pho_p + self.pho_n)/2 * 100
        th_l = np.percentile(losses, 100-noise_rate*fraction)
        th_c = np.percentile(conf, noise_rate*fraction)

        drop_indices = np.argwhere(np.logical_and(
            losses > th_l, conf < th_c)).reshape(-1)
        self.drop(drop_indices)
        drop_indices_t = torch.from_numpy(drop_indices)
        noise = (torch.sign(self.target_tensor)[drop_indices_t]
                 != self.label_tensor[drop_indices_t])
        print('noise/drop: {}/{}'.format(torch.sum(noise), len(drop_indices)))

    def remove_no_effect(self):
        if torch.sum(self.target_tensor == 0) == 0:
            return None, None
        remove_indices = torch.from_numpy(np.argwhere(
            (self.target_tensor == 0).numpy().reshape(-1)).reshape(-1))
        remain_indices = torch.from_numpy(np.argwhere(
            (self.target_tensor != 0).numpy().reshape(-1)).reshape(-1))
        removed_data = self.data_tensor[remove_indices]
        removed_labels = self.label_tensor[remove_indices]
        self.data_tensor = self.data_tensor[remain_indices]
        self.label_tensor = self.label_tensor[remain_indices]
        self.weight_tensor = self.weight_tensor[remain_indices]
        self.target_tensor = self.target_tensor[remain_indices]
        self.is_used_tensor = self.is_used_tensor[remain_indices]
        return removed_data, removed_labels


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


def datasets_initialization(
        data, labels, init_size, init_weight, pho_p=0, pho_n=0):
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
        init_weight * torch.ones(init_size, 1), pho_p, pho_n)
    return unlabeled_set, labeled_set


def datasets_initialization_kcenter(
        data, labels, init_size, init_weight, pho_p=0, pho_n=0):
    if torch.is_tensor(data):
        data = data.numpy()
    if torch.is_tensor(labels):
        labels = labels.numpy()
    kcenter = KCenter(data, labels)
    for i in range(init_size-1):
        kcenter.select_one()
    unlabeled_set = torch.utils.data.TensorDataset(
        torch.from_numpy(kcenter.pool).float(),
        torch.from_numpy(kcenter.pool_y).float())
    labeled_set = WeightedTensorDataset(
        torch.from_numpy(kcenter.selected).float(),
        torch.from_numpy(kcenter.selected_y).float(),
        init_weight * torch.ones(init_size, 1), pho_p, pho_n)
    return unlabeled_set, labeled_set
