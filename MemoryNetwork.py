import numpy as np
import random as R
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
import torch
from scipy import spatial

from utils import KL_divergence

def random_uniform(shape, low, high):
    x = torch.rand(*shape)
    result = (high - low) * x + low
    # print('initial K shape (before normalize):', result.shape)

    return result

class MemoryNetwork:
    def __init__(self, memory_size, spatial_dim, top_k=1):
        self.memory_size = memory_size
        self.K = F.normalize(random_uniform((self.memory_size, spatial_dim), -0.01, 0.01), dim=1)
        # print('initial K shape:', self.K.shape)
        self.V = F.normalize(random_uniform((self.memory_size, 313), 0, 0.01), p = 1, dim=1)
        self.A = np.array([0 for _ in range(self.memory_size)])
        self.top_k = top_k

    def update(self, q, v, threshold, idx):
        n = self.get_neighbors(q, self.top_k)
        top1_indeces = list(n[:, 0])
        # print('====neighbors shape:', n.shape)
        # print('====top1_indeces shape:', top1_indeces)
        # print('====V shape:', self.V.shape)
        self.A = self.A + 1 # numpy array doesn't support +=

        similarity = KL_divergence(self.V[top1_indeces], v)
        print('FFF similarity:', similarity.shape, similarity)
        case1 = np.where(similarity < threshold)
        case2 = np.where(similarity >= threshold)

        # case 1 -> query and memory slot are within the same class
        # so update a memory slot to be an average
        print('FFF case1:', len(case1), case1)
        print('FFF case2:', len(case2), case2)
        print('FFF K1 shape:', self.K[case1].shape)
        print('FFF q1 shape:', q[case1].shape)
        # print('FFF K2 shape:', self.K[case2].shape, self.K[case2])
        print('FFF q2 shape:', q[case2].shape)
        if q[case1].shape[0]:
            self.K[case1] = (q[case1] + self.K[case1]) / np.sqrt(np.power(q[case1] + self.K[case1], 2))
            self.A[case1] = 0

        # case 2 otherwise
        if q[case2].shape[0]:
            r = self.pick_the_oldest(len(case2))
            self.K[r] = q[case2]
            self.V[r] = v[case2]
            self.A[r] = 0

    def pick_the_oldest(self, num2pick):
        age_values = [np.array([i, age]) for i, age in enumerate(self.A)]
        age_values.sort(key=lambda x: x[1])
        return np.array(age_values)[:num2pick, 0]

    def dist(self, q):
        # print('q shape:', np.squeeze(q.reshape(-1, 1)).shape)
        # print('K shape:', self.K[k.long()].shape)
        # print('K shape:', np.squeeze(k.reshape(-1, 1)).shape)
        return np.array([spatial.distance.cosine(np.squeeze(q.reshape(-1, 1)),
                                        np.squeeze(k.reshape(-1, 1)))
                                        for k in self.K])

    def get_neighbors(self, q, num_neighbors=1):
        distances = []
        for q_ in q:
            q_dist = self.dist(q_)
            # print('### distances:', q_dist.shape, q_dist)
            to_add = np.array(q_dist.argsort())[-num_neighbors:][::-1]
            # print('appending', to_add.shape, to_add)
            distances.append(np.array(q_dist.argsort())[-num_neighbors:][::-1])
        distances = np.array(distances)
        return distances
