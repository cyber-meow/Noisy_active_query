import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn import datasets
from copy import deepcopy


n_positive = 10000
n_negative = 10000
n = n_positive + n_negative

init_size = 90

pho_p = 0.5
pho_n = 0
pho_p_c = pho_p
pho_n_c = pho_n


incr_size = 6
incr_pool_size = 1500
incr_times = 8
convex_epochs = 1000
retrain_epochs = 12000


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
    H = 6
    output_size = 1
    idd = 0

    def __init__(self, pho_p=0, pho_n=0, lr=2e-3):
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
        loss = self.basic_loss(targets*outputs, typ=typ)
        loss = (1-pho_ny+pho_y) * loss
        # logistic_y = self.basic_loss(targets*outputs, typ=typ)
        # logistic_ny = self.basic_loss(-targets*outputs, typ=typ)
        # loss = (1-pho_ny)*logistic_y - pho_y*logistic_ny
        if ws is not None:
            assert ws.shape == loss.shape
            loss *= ws
        total_loss = torch.sum(loss)/(1-self.pho_n-self.pho_p)
        return loss, total_loss

    def train(self, x_train, y_train, num_epochs, convex_epochs, ws=None):
        self.training_points.extend(x_train)
        inputs = Variable(torch.from_numpy(x_train)).float()
        targets = Variable(torch.from_numpy(y_train)).view(-1, 1).float()
        if ws is not None:
            ws = Variable(torch.from_numpy(ws).float(), requires_grad=False)
        for i in range(num_epochs):
            if i < convex_epochs:
                loss = self.train_step(inputs, targets, ws=ws, typ='lg')
            else:
                loss = self.train_step(inputs, targets, ws=ws, typ='sig')
            loss = self.train_step(inputs, targets, ws=ws)
            if i % 2000 == 0:
                print('Epoch {}, Loss: {:.4f}'.format(i, loss.data[0]))
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


def bootstrap(xs, ys, m):
    n = xs.shape[0]
    idxs = np.random.choice(n, size=m, replace=True)
    return xs[idxs], ys[idxs]


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

training_xs = x_init
training_ys = y_init

tr_x = training_xs
tr_y = training_ys

cls = NoisyLearning(pho_p=pho_p_c, pho_n=pho_n_c)
cls2 = deepcopy(cls)
cm = plt.get_cmap('gist_rainbow')
cm2 = plt.get_cmap('gist_earth')

cont = None
cont2 = None


for trial in range(incr_times+1):

    cls = NoisyLearning(pho_p=pho_p_c, pho_n=pho_n_c)
    cls2 = deepcopy(cls)

    print('classifier 1')
    cls.train(training_xs, training_ys, retrain_epochs, convex_epochs)
    accuracy = cls.get_accuracy(x_test, y_test)
    if trial >= 1:
        cont.collections[0].set_linewidth(1)
        cont.collections[0].set_alpha(0.3)
    cont = cls.model.plot_boundary(ax, colors=[cm(trial/incr_times)])
    print(accuracy)
    plt.pause(0.05)

    # print('classifier 2')
    # cls2.train(tr_x, tr_y, retrain_epochs, convex_epochs)
    # accuracy = cls2.get_accuracy(x_test, y_test)
    # if trial >= 1:
    #     cont2.collections[0].set_linewidth(1)
    #     cont2.collections[0].set_alpha(0.3)
    # cont2 = cls2.model.plot_boundary(ax, colors=[cm2(trial/incr_times)])
    # print(accuracy)
    # plt.pause(0.05)

    outputs = cls.model(Variable(torch.from_numpy(x_all)).float())
    sigmoid = nn.Sigmoid()
    probs = sigmoid(outputs).data.numpy().reshape(-1)
    # idxs = np.argpartition(np.abs(probs-0.5), incr_size)[:incr_size]
    s_idxs = np.argsort(np.abs(probs-0.5))[:incr_pool_size]
    drawn = np.random.choice(s_idxs, incr_size, replace=False)

    x_selected = x_all[drawn]
    y_selected = y_all_corrupted[drawn]

    idxs = np.ones(len(x_all))
    idxs[drawn] = False
    idxs = np.argwhere(idxs).reshape(-1)
    x_all = x_all[idxs]
    y_all_corrupted = y_all_corrupted[idxs]

    # outputs = cls.model(Variable(torch.from_numpy(training_xs)).float())
    # sigmoid = nn.Sigmoid()
    # probs = sigmoid(outputs).data.numpy().reshape(-1)
    # small_margin = np.argsort(np.abs(probs-0.5))[:int(len(training_xs)/6)]

    # idxs = np.ones(len(training_xs))
    # idxs[small_margin] = False
    # idxs = np.logical_and(idxs, probs >= 0.5)
    # idxs = np.argwhere(idxs).reshape(-1)
    # tr_x = training_xs[idxs]
    # tr_y = training_ys[idxs]

    sx, sy = x_selected.T
    plt.scatter(sx, sy, s=7, label='{}'.format(trial))
    sx, sy = x_selected[y_selected == -1].T
    plt.scatter(sx, sy, s=25, c='black', alpha=0.2)
    # plt.legend()
    plt.pause(0.05)

    training_xs = np.concatenate([training_xs, x_selected])
    training_ys = np.concatenate([training_ys, y_selected])

    # tr_x = np.concatenate([tr_x, x_selected])
    # tr_y = np.concatenate([tr_y, y_selected])


while not plt.waitforbuttonpress(1):
    pass
