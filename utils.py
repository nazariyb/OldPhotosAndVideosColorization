import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle


def recreate_image(codebook, labels, w, h, n_colors):
    d = codebook.shape[1]
    image = np.zeros((w, h, n_colors))
    label_idx = 0
    channels = [ch[0] * ch[1] for ch in codebook]
    for i in range(w):
        for j in range(h):
            value = codebook[labels[label_idx]]
            image[i][j][channels.index(value[0] * value[1])] = 1
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
    return recreate_image(kmeans.cluster_centers_, labels, w, h, n_colors).astype(np.uint8)


def KL_divergence(V, v, dim=1):
    V = V.cpu()
    v = v.cpu()
    return np.array([np.sum(np.where(Vv != 0, Vv * np.log(Vv / vv), dim))
                    for Vv, vv in zip(V, v)])