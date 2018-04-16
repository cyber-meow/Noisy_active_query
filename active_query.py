import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import settings
from scipy.spatial import distance


class ActiveQuery(object):

    def update(self, unlabeled_set, labeled_set, drawn, weights):
        labeled_set.update(
            unlabeled_set.data_tensor[drawn],
            unlabeled_set.target_tensor[drawn],
            weights)
        idxs = np.ones(len(unlabeled_set))
        idxs[drawn] = False
        idxs = torch.from_numpy(np.argwhere(idxs).reshape(-1))
        unlabeled_set.data_tensor = unlabeled_set.data_tensor[idxs]
        unlabeled_set.target_tensor = unlabeled_set.target_tensor[idxs]

    def query(self, labeled_set, unlabeled_set, k, *args):
        raise NotImplementedError


class RandomQuery(ActiveQuery):

    def query(self, unlabeled_set, labeled_set, k, unit_weight):
        drawn = torch.from_numpy(
            np.random.choice(len(unlabeled_set), k, replace=False))
        x_selected = unlabeled_set.data_tensor[drawn]
        y_selected = unlabeled_set.target_tensor[drawn]
        self.update(
            unlabeled_set, labeled_set, drawn,
            unit_weight*torch.ones(k, 1))
        return x_selected, y_selected, unit_weight*torch.ones(k, 1)


class UncertaintyQuery(ActiveQuery):

    def query(self, unlabeled_set, labeled_set,
              k, cls, incr_pool_size, unit_weight):
        output = cls.model(
            Variable(unlabeled_set.data_tensor)).type(settings.dtype)
        sigmoid = nn.Sigmoid()
        probs = sigmoid(output).data.numpy().reshape(-1)
        s_idxs = np.argsort(np.abs(probs-0.5))[:incr_pool_size]
        drawn = torch.from_numpy(np.random.choice(s_idxs, k, replace=False))
        x_selected = unlabeled_set.data_tensor[drawn]
        y_selected = unlabeled_set.target_tensor[drawn]
        self.update(
            unlabeled_set, labeled_set, drawn,
            unit_weight*torch.ones(k, 1))
        return x_selected, y_selected, unit_weight*torch.ones(k, 1)


class IWALQuery(ActiveQuery):

    def __init__(self):
        self.weight_factor = None

    def query(self, unlabeled_set, labeled_set, k, clss, weight_ratio=None):

        n = len(unlabeled_set)

        min_ls_p = np.inf * np.ones([n, 1])
        max_ls_p = -np.inf * np.ones([n, 1])
        min_ls_n = np.inf * np.ones([n, 1])
        max_ls_n = -np.inf * np.ones([n, 1])
        p_predict = np.zeros(n)
        n_predict = np.zeros(n)

        for cls in clss:
            output = cls.model(
                Variable(unlabeled_set.data_tensor).type(settings.dtype)).cpu()
            predict = torch.sign(output).data.numpy().reshape(-1)
            p_predict = np.logical_or(predict == 1, p_predict)
            n_predict = np.logical_or(predict == -1, n_predict)
            loss_p, _ = cls.compute_loss(
                output, Variable(torch.ones(n, 1).float()))
            loss_n, _ = cls.compute_loss(
                output, Variable(-torch.ones(n, 1).float()))
            min_ls_p = np.minimum(min_ls_p, loss_p.data.numpy())
            max_ls_p = np.maximum(max_ls_p, loss_p.data.numpy())
            min_ls_n = np.minimum(min_ls_n, loss_n.data.numpy())
            max_ls_n = np.maximum(max_ls_n, loss_n.data.numpy())

        ls_diffs_p = (max_ls_p-min_ls_p).reshape(-1)
        ls_diffs_n = (max_ls_n-min_ls_n).reshape(-1)
        disagreement_area = np.logical_and(p_predict, n_predict)
        ls_diffs = np.maximum(ls_diffs_p, ls_diffs_n) * disagreement_area
        ls_sum = np.sum(ls_diffs)
        sampling_probs = np.minimum(k*ls_diffs/ls_sum, 1)

        print(np.min(
            sampling_probs[sampling_probs != 0]), np.max(sampling_probs))

        drawn = np.random.binomial(
            np.ones(n, dtype=int), sampling_probs).astype(bool)
        drawn = torch.from_numpy(np.argwhere(drawn).reshape(-1))
        weights = 1/sampling_probs[drawn].reshape(-1, 1)

        if weight_ratio is not None:
            if self.weight_factor is None:
                avg_weight = np.mean(weights)
                self.weight_factor = weight_ratio/avg_weight
            weights *= self.weight_factor

        weights = torch.from_numpy(weights).float()
        print(weights)

        x_selected = unlabeled_set.data_tensor[drawn]
        y_selected = unlabeled_set.target_tensor[drawn]
        self.update(unlabeled_set, labeled_set, drawn, weights)

        return x_selected, y_selected, weights


class HeuristicRelabel(object):

    @staticmethod
    def confidence_scores(data, votes, k, pho_p, pho_n):
        p_d = distance.squareform(distance.pdist(data))
        sigma = np.mean(np.sort(p_d, axis=1)[:, :k])
        K = np.exp(-p_d**2/sigma**2)
        votes = votes.reshape(-1)
        score = np.sum(K * votes, axis=1) * votes
        score = 2*score/np.std(score)
        conf = 1/(1+np.exp(-score))
        return conf

    def diverse_flipped(self, labeled_set, num_clss, k, kn, pho_p, pho_n):
        conf = self.confidence_scores(
            labeled_set.data_tensor.numpy().reshape(len(labeled_set), -1),
            labeled_set.target_tensor.numpy(), kn, pho_p, pho_n)
        drop_indices = np.argwhere(conf < 0.2).reshape(-1)
        print(drop_indices)
        labeled_set.drop(drop_indices)
        possible_query_indices = np.logical_and(0.2 <= conf, conf < 0.6)
        possible_query_indices = np.argwhere(
            possible_query_indices).reshape(-1)
        indices_datasets = []
        for _ in range(num_clss):
            if k <= len(possible_query_indices):
                flipped_idxs = np.random.choice(
                    possible_query_indices, k, replace=False)
            else:
                flipped_idxs = possible_query_indices
            new_set = labeled_set.modify(flipped_idxs)
            indices_datasets.append((flipped_idxs, new_set))
        return indices_datasets, drop_indices
