import torch
from torch.autograd import Variable
import numpy as np
import settings


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
        self.update(
            unlabeled_set, labeled_set, drawn,
            unit_weight*torch.ones(k, 1))


class IWALQuery(ActiveQuery):

    def query(self, unlabeled_set, labeled_set, k, clss):

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
        weights = torch.from_numpy(
            1/sampling_probs[drawn].reshape(-1, 1)).float()
        print(weights)

        self.update(unlabeled_set, labeled_set, drawn, weights)
