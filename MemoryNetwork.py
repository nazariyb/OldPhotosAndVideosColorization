import numpy as np
import random as R
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import functional as F
import torch
from scipy import spatial

from utils import KL_divergence
from device2use import device

def random_uniform(shape, low, high):
    x = torch.rand(*shape)
    result = (high - low) * x + low

    return result

class MemoryNetwork:
    def __init__(self, memory_size, spatial_dim, top_k=1):
        self.memory_size = memory_size
        self.K = F.normalize(random_uniform((self.memory_size, spatial_dim), -0.01, 0.01), dim=1).to(device)
        self.V = F.normalize(random_uniform((self.memory_size, 313), 0, 0.01), p = 1, dim=1).to(device)
        self.A = torch.from_numpy(np.array([0 for _ in range(self.memory_size)])).to(device)
        self.top_k = top_k

    def update(self, q, v, threshold, idx):
        n = self.get_neighbors(q, self.top_k)
        top1_indeces = list(n[:, 0])
        self.A = self.A + 1 # numpy array doesn't support +=

        similarity = KL_divergence(self.V[top1_indeces], v)
        case1 = np.where(similarity < threshold)
        case2 = np.where(similarity >= threshold)

        self.K = self.K.cpu()
        self.V = self.V.cpu()
        self.A = self.A.cpu()
        q = q.cpu()
        v = v.cpu()

        # case 1 -> query and memory slot are within the same class
        # so update a memory slot to be an average
        if case1[0].shape[0] > 0:
            self.K[case1] = (q[case1] + self.K[case1]) / np.sqrt(np.power(q[case1] + self.K[case1], 2))
            self.A[case1] = 0

        # case 2 otherwise
        if case2[0].shape[0] > 0:
            r = self.pick_the_oldest(case2[0].shape[0])
            self.K[r] = q[case2]
            self.V[r] = v[case2]
            self.A[r] = 0

        self.K = self.K.to(device)
        self.V = self.V.to(device)
        self.A = self.A.to(device)
        q = q.to(device)
        v = v.to(device)

    def pick_the_oldest(self, num2pick):
        age_values = [np.array([i, age]) for i, age in enumerate(self.A)]
        age_values.sort(key=lambda x: x[1])
        return np.array(age_values)[:num2pick, 0]

    def dist(self, q):
        return np.array([spatial.distance.cosine(np.squeeze(q.cpu().reshape(-1, 1)),
                                        np.squeeze(k.reshape(-1, 1)))
                                        for k in self.K.cpu()])

    def get_neighbors(self, q, num_neighbors=1):
        distances = []
        for q_ in q:
            q_dist = self.dist(q_)
            to_add = np.array(q_dist.argsort())[-num_neighbors:][::-1]
            distances.append(np.array(q_dist.argsort())[-num_neighbors:][::-1])
        distances = np.array(distances)
        return distances
