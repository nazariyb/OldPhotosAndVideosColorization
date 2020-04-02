import numpy as np
import random as R
from sklearn.metrics.pairwise import cosine_similarity

from utils import KL_divergence

class MemoryNetwork:
    def __init__(self, memory_size, top_k=256):
        self.memory_size = memory_size
        self.K = np.array([None for _ in range(self.memory_size)])
        self.V = np.array([None for _ in range(self.memory_size)])
        self.A = np.array([0 for _ in range(self.memory_size)])
        self.top_k = top_k

    def update(self, q, v, threshold, idx):
        n = self.get_neighbors(q, self.top_k)
        top1_indeces = n[0]

        self.age = self.age + 1

        similarity = KL_divergence(self.V[top1_indeces], v)
        case1 = np.where(similarity < threshold)
        case2 = np.where(similarity >= threshold)

        # case 1 -> query and memory slot are within the same class
        # so update a memory slot to be an average
        self.K[case1] = (q + self.K[case1]) / np.sqrt(np.power(q[case1] + self.K[case1], 2))
        self.A[case1] = 0

        r = self.pick_the_oldest(len(case2))
        self.K[r] = q[case2]
        self.V[r] = v[case2]
        self.A[r] = 0

    def pick_the_oldest(self, num2pick):
        age_values = [np.array([i, age]) for i, age in enumerate(self.A)]
        age_values.sort(key=lambda x: x[1])
        return np.array(age_values)[:num2pick, 0]

    def dist(self, q, k):
        return cosine_similarity(q, self.K[k])

    def get_neighbors(self, q, num_neighbors=1):
        distances = []
        for i, k in enumerate(self.K):
            dist = self.dist(q[i], k)
            distances.append(np.array([i, dist]))
        distances.sort(key=lambda x: x[1])
        distances = np.array(distances)
        return distances[:num_neighbors, 0]