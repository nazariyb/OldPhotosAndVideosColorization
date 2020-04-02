import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle


def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            # image[i][j] = np.array(list(reversed(codebook[labels[label_idx]])))
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


def quantize_colors(img, n_colors):
    img = np.array(img, dtype=np.float64)

    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0)[:10_000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)

    return recreate_image(kmeans.cluster_centers_, labels, w, h).astype(np.uint8)


def KL_divergence(V, v, dim=1):
    return np.sum(np.where(V != 0, V * np.log(V / v), dim))