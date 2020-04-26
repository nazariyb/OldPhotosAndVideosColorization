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
    print('channels:', len(channels))
    for i in range(w):
        for j in range(h):
            # image[i][j] = np.array(list(reversed(codebook[labels[label_idx]])))
            value = codebook[labels[label_idx]]
            # print('//// set up a value:', value)
            image[i][j][channels.index(value[0] * value[1])] = 1
            label_idx += 1
    # print('example1:', image[:, :, 0])
    # print('example2:', image[:, :, 1])
    return image


def quantize_colors(img, n_colors):
    img = np.array(img, dtype=np.float64)
    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))
    print('//// quantizing', image_array.shape)

    image_array_sample = shuffle(image_array, random_state=0)[:10_000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    print('//// quantized', recreate_image(kmeans.cluster_centers_, labels, w, h, n_colors).astype(np.uint8).shape)
    return recreate_image(kmeans.cluster_centers_, labels, w, h, n_colors).astype(np.uint8)


def KL_divergence(V, v, dim=1):
    print('KLKLK   V', V.shape)
    print('KLKLK   v', v.shape)
    return np.array([np.sum(np.where(Vv != 0, Vv * np.log(Vv / vv), dim))
                    for Vv, vv in zip(V, v)])