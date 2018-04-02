import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn import datasets


n_positive = 10000
n_negative = 10000
n = n_positive + n_negative

init_size = 90
bootstrap_size = 70
bootstrap_ratio = 0.7
init_w = 300

pho_p = 0.4
pho_n = 0.1
pho_p_c = pho_p
pho_n_c = pho_n

num_clss = 6
incr_size = 8
incr_times = 7
retrain_epochs = 12000
num_epochs = 16000
final_epochs = 20000


class TwoLayerNet(nn.Module):

    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def border_func(self, x, y):
        inp = Variable(torch.from_numpy(np.array([x, y])).float())
        return self.forward(inp).data.numpy()

    def plot_boundary(self, ax, **kwargs):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xs = np.linspace(xmin, xmax, 100)
        ys = np.linspace(ymin, ymax, 100)
        xv, yv = np.meshgrid(xs, ys)
        border_func = np.vectorize(self.border_func)
        cont = plt.contour(xv, yv, border_func(xv, yv), [0], **kwargs)
        return cont


class NoisyLearning(object):

    input_size = 2
    output_size = 1
    H = 10
    idd = 0

    def __init__(self, pho_p=0, pho_n=0, lr=5e-3):
        self.model = TwoLayerNet(self.input_size, self.H, self.output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.training_points = []
        assert(pho_p + pho_n < 1)
        self.pho_p = pho_p
        self.pho_n = pho_n
        self.idd = NoisyLearning.idd
        NoisyLearning.idd += 1

    def basic_loss(self, fx, typ='lg'):
        if typ == 'lg':
            negative_logistic = nn.LogSigmoid()
            return -negative_logistic(fx)
        if typ == 'ramp':
            return torch.clamp((1-fx)/2, 0, 1)
        if typ == 'sig':
            sigmoid = nn.Sigmoid()
            return sigmoid(-fx)

    def train_step(self, inputs, targets, ws=None, typ='lg'):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        _, loss = self.compute_loss(outputs, targets, ws=ws, typ=typ)
        loss.backward()
        self.optimizer.step()
        return loss

    def compute_loss(self, outputs, targets, ws=None, typ='lg'):
        a = (self.pho_p - self.pho_n)/2
        b = (self.pho_p + self.pho_n)/2
        pho_y = a * targets + b
        pho_ny = self.pho_p + self.pho_n - pho_y
        logistic_y = self.basic_loss(targets*outputs, typ=typ)
        logistic_ny = self.basic_loss(-targets*outputs, typ=typ)
        loss = (1-pho_ny)*logistic_y - pho_y*logistic_ny
        if ws is not None:
            assert ws.shape == loss.shape
            loss *= ws
        total_loss = torch.sum(loss)/(1-self.pho_n-self.pho_p)
        return loss, total_loss

    def train(self, x_train, y_train, num_epochs,
              ws=None, typ='sig', pri=False):
        self.training_points.extend(x_train)
        inputs = Variable(torch.from_numpy(x_train)).float()
        targets = Variable(torch.from_numpy(y_train)).view(-1, 1).float()
        if ws is not None:
            ws = Variable(torch.from_numpy(ws).float(), requires_grad=False)
        for i in range(num_epochs):
            loss = self.train_step(inputs, targets, ws=ws, typ=typ)
            if pri and i % 2000 == 0:
                print('Epoch {}, Loss: {:.4f}'.format(i, loss.data[0]))
                # self.model.plot_boundary(ax)
                # plt.pause(0.05)
        print(self.idd)
        print('Epoch {}, Loss: {:.4f}'.format(i, loss.data[0]))

    def predict(self, inputs):
        outputs = self.model(inputs)
        return torch.sign(outputs)

    def get_accuracy(self, x_test, y_test):
        inputs = Variable(torch.from_numpy(x_test)).float()
        return np.sum(
            self.predict(inputs).data.numpy() == y_test.reshape(-1, 1)) / n

    def plot_training_points(self, **kwargs):
        tx, ty = np.array(self.training_points).T
        plt.scatter(tx, ty, **kwargs)


def bootstrap(xs, ys, ws, m):
    n = xs.shape[0]
    idxs = np.random.choice(n, size=m, replace=True)
    return xs[idxs], ys[idxs], ws[idxs]


def new_classifs(kept_cls, n, lr=2e-3):
    clss = [kept_cls]
    for _ in range(n-1):
        clss.append(NoisyLearning(pho_p=pho_p_c, pho_n=pho_n_c, lr=lr))
    return clss


def best_classif(clss, tr_x, tr_y, tr_w=None, typ='sig'):
    best = None
    best_loss = np.inf
    tr_x = Variable(torch.from_numpy(tr_x)).float()
    tr_y = Variable(torch.from_numpy(tr_y)).float().view(-1, 1)
    if tr_w is not None:
        tr_w = Variable(torch.from_numpy(tr_w).float(), requires_grad=False)
    for cls in clss:
        outputs = cls.model(tr_x)
        _, total_loss = cls.compute_loss(outputs, tr_y, ws=tr_w, typ=typ)
        total_loss = total_loss.data.numpy()
        if total_loss < best_loss:
            best_loss = total_loss
            best = cls
    return best


x_all, y_all = datasets.make_moons(n, shuffle=False, noise=0.07)
y_all = y_all*2 - 1

nidxs = np.random.permutation(n_negative)
pidxs = n_negative + np.random.permutation(n_positive)
x_all = x_all[np.concatenate([nidxs, pidxs])]

negative_samples = x_all[:n_negative]
positive_samples = x_all[n_negative:]

fig, ax = plt.subplots()

plt.ion()
plt.show()

px, py = np.array(positive_samples).T
nx, ny = np.array(negative_samples).T
plt.scatter(px, py, color='mistyrose', s=3)
plt.scatter(nx, ny, color='turquoise', s=3)
plt.pause(0.05)

n_pcorrupted = np.random.binomial(n_positive, pho_p)
n_ncorrupted = np.random.binomial(n_negative, pho_n)

y_all_corrupted = np.r_[
    -np.ones(n_negative-n_ncorrupted),
    np.ones(n_ncorrupted+n_positive-n_pcorrupted),
    -np.ones(n_pcorrupted)]

x_test, y_test = datasets.make_moons(n, shuffle=False, noise=0.07)
y_test = y_test*2 - 1

drawn = np.random.choice(n, init_size, replace=False)
x_init = x_all[drawn]
y_init = y_all_corrupted[drawn]

cx, cy = np.array(x_init).T
plt.scatter(cx, cy, s=3, color='yellow')
cx, cy = x_init[y_init == -1].T
plt.scatter(cx, cy, s=3, c='black', alpha=0.2)
plt.pause(0.05)

x_all_v = Variable(torch.from_numpy(x_all)).float()


tr_x = x_init
tr_y = y_init
tr_w = init_w * np.ones([init_size, 1])


kept_cls = NoisyLearning(pho_p=pho_p_c, pho_n=pho_n_c, lr=2e-3)

conts = []
cm = plt.get_cmap('gist_rainbow')


for trial in range(incr_times):

    clss = new_classifs(kept_cls, num_clss)

    min_ls_p = np.inf * np.ones([n, 1])
    max_ls_p = -np.inf * np.ones([n, 1])
    min_ls_n = np.inf * np.ones([n, 1])
    max_ls_n = -np.inf * np.ones([n, 1])

    p_predict = np.zeros(n)
    n_predict = np.zeros(n)

    for i, cls in enumerate(clss):
        # bootstrap_size = int(tr_x.shape[0]*bootstrap_ratio)
        xs, ys, ws = bootstrap(tr_x, tr_y, tr_w, bootstrap_size)
        cls.train(xs, ys, retrain_epochs, ws=ws)
        accuracy = cls.get_accuracy(x_test, y_test)
        print(accuracy)
        if trial >= 1:
            for coll in conts[0].collections:
                coll.remove()
            del conts[0]
        conts.append(cls.model.plot_boundary(ax, colors=[cm(i/num_clss)]))
        plt.pause(0.05)
        outputs = cls.model(x_all_v)
        prt = torch.sign(outputs).data.numpy().reshape(-1)
        p_predict = np.logical_or(prt == 1, p_predict)
        n_predict = np.logical_or(prt == -1, n_predict)
        loss_p, _ = cls.compute_loss(
            outputs, Variable(torch.ones(n, 1).float()), typ='sig')
        loss_n, _ = cls.compute_loss(
            outputs, Variable(-torch.ones(n, 1).float()), typ='sig')
        # loss_p = np.maximum(loss.data.numpy(), 0)
        min_ls_p = np.minimum(min_ls_p, loss_p.data.numpy())
        max_ls_p = np.maximum(max_ls_p, loss_p.data.numpy())
        min_ls_n = np.minimum(min_ls_n, loss_n.data.numpy())
        max_ls_n = np.maximum(max_ls_n, loss_n.data.numpy())

    ls_diffs_p = (max_ls_p-min_ls_p).reshape(-1) * p_predict
    ls_diffs_n = (max_ls_n-min_ls_n).reshape(-1) * n_predict
    ls_diffs = np.maximum(ls_diffs_p, ls_diffs_n)
    # ls_diffs = np.ones(pool_size)
    ls_sum = np.sum(ls_diffs)
    sampling_probs = np.minimum(incr_size*ls_diffs/ls_sum, 1)
    # idx_largests = np.argpartition(ls_diffs, -incr_size)[-incr_size:]
    # x_selected = x_all[idx_largests]
    # y_selected = y_all_corrupted[idx_largests]

    print(np.min(sampling_probs), np.max(sampling_probs))

    to_draw = np.random.binomial(
        np.ones(n, dtype=int), sampling_probs).astype(bool)
    x_selected = x_all[to_draw]
    y_selected = y_all_corrupted[to_draw]
    new_weights = (1/sampling_probs)[to_draw].reshape(-1, 1)
    print(new_weights)
    # new_weights = init_w * np.ones([incr_size, 1])

    sx, sy = x_selected.T
    plt.scatter(sx, sy, s=10, label='{}'.format(trial))
    sx, sy = x_selected[y_selected == -1].T
    plt.scatter(sx, sy, s=25, c='black', alpha=0.2)
    plt.legend()
    plt.pause(0.05)

    tr_x = np.concatenate([tr_x, x_selected])
    tr_y = np.concatenate([tr_y, y_selected])
    tr_w = np.concatenate([tr_w, new_weights])
    bootstrap_size += max(np.shape(x_selected)[0]-1, 0)

    kept_cls = best_classif(clss, tr_x, tr_y, tr_w=tr_w, typ='sig')


if incr_times == 0:
    for i in range(num_clss):
        cls = NoisyLearning(pho_p=pho_p_c, pho_n=pho_n_c)
        cls.train(tr_x, tr_y, num_epochs, ws=tr_w, typ='sig', pri=True)
        cls.model.plot_boundary(ax, colors=[cm(i/num_clss)])
        plt.scatter([-1], [-0.5], s=1, c=cm(i/num_clss), label='{}'.format(i))
        plt.legend()
        plt.pause(0.05)
        accuracy = cls.get_accuracy(x_test, y_test)
        print(accuracy)

else:
    cls = NoisyLearning(pho_p=pho_p_c, pho_n=pho_n_c)
    print(cls.idd)
    cls.train(tr_x, tr_y, final_epochs, ws=tr_w, typ='sig')
    cls.model.plot_boundary(ax, colors=['black'])
    plt.pause(0.05)
    accuracy = cls.get_accuracy(x_test, y_test)
    print(accuracy)


while not plt.waitforbuttonpress(1):
    pass
