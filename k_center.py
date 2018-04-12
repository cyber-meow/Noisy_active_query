import numpy as np
from scipy.spatial import distance


class KCenter(object):

    def __init__(self, pool, pool_y, selected=None, selected_y=None):
        self.pool = pool
        self.pool_y = pool_y
        self.selected = selected
        self.selected_y = selected_y
        if self.selected is None:
            pid = np.random.choice(self.pool.shape[0])
            self.selected = np.array([self.pool[pid]])
            self.selected_y = np.array([self.pool_y[pid]])
            self.pool = np.delete(self.pool, pid, axis=0)
            self.pool_reshape = self.pool.reshape(len(self.pool), -1)
            self.pool_y = np.delete(self.pool_y, pid, axis=0)
        self.init_distance()

    def init_distance(self):
        distances = distance.cdist(
            self.pool_reshape,
            self.selected.reshape(len(self.selected), -1),
            metric='minkowski', p=1)
        self.pool_min_distances = np.min(distances, axis=1).reshape(-1, 1)

    def update_point(self, pid):
        self.pool_min_distances = np.delete(
            self.pool_min_distances, pid, axis=0)
        self.selected = np.append(
            self.selected, [self.pool[pid]], axis=0)
        self.selected_y = np.append(
            self.selected_y, [self.pool_y[pid]], axis=0)
        self.pool = np.delete(self.pool, pid, axis=0)
        self.pool_reshape = np.delete(self.pool_reshape, pid, axis=0)
        self.pool_y = np.delete(self.pool_y, pid, axis=0)

    def update_distance(self, point):
        distances = distance.cdist(
            self.pool_reshape, [point.reshape(-1)],
            metric='minkowski', p=1)
        self.pool_min_distances = np.minimum(
            self.pool_min_distances, distances)

    def select_one(self):
        pid = np.argmax(self.pool_min_distances)
        point = self.pool[pid]
        self.update_point(pid)
        self.update_distance(point)
