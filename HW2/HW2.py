import math
import warnings
import cv2
import numpy as np
from numpy import uint8

warnings.filterwarnings('ignore')


def padding(fmap, filter_shape):
    for i in range(filter_shape[0]):
        W1, H1 = fmap.shape[:2]
        fmap = np.insert(fmap, 0, 0, axis=0)
        fmap = np.insert(fmap, W1 + 1, 0, axis=0)
    for i in range(filter_shape[1]):
        W1, H1 = fmap.shape[:2]
        fmap = np.insert(fmap, 0, 0, axis=1)
        fmap = np.insert(fmap, H1 + 1, 0, axis=1)
    return fmap


def relu(x):
    if x < 0:
        x = 0
    if x > 255:
        x = 255
    return x


def conv(input_fmap, kernels, shift, stride):
    W1, H1, C1 = input_fmap.shape
    M, C2, S, R = kernels.shape
    k1_del = S // 2
    k2_del = R // 2
    input_fmap = padding(input_fmap, [k1_del, k2_del])
    W2 = round((W1 - S) / stride + 1)
    H2 = round((H1 - R) / stride + 1)
    output_fmap = np.zeros((W2, H2, M), dtype=float)
    for m in range(M):
        for w in range(W2):
            for h in range(H2):
                output_fmap[w, h, m] = shift[m]
                for i in range(S):
                    for j in range(R):
                        for k in range(C2):
                            output_fmap[w, h, m] += (input_fmap[stride * w + i, stride * h + j, k] *
                                                     kernels[m, k, i, j]).sum()
                output_fmap[w, h, m] = relu(output_fmap[w, h, m])
    return np.asarray(output_fmap, dtype=uint8)


def normalize(fmap):
    beta = np.zeros(fmap.shape[0], dtype=float)
    gamma = np.ones(fmap.shape[0], dtype=float)
    mean = np.zeros(fmap.shape[1], dtype=float)
    eps = 0.000000001
    for j in range(fmap.shape[1]):
        mean[j] = np.mean(fmap[:, j])
        for i in range(fmap.shape[0]):
            fmap[i, j] = ((fmap[i, j] - mean[j]) / math.sqrt(np.std(fmap[:, j]) + eps))
            fmap[i, j] = gamma[i] * fmap[i, j] + beta[i]
    return fmap


def max_pooling(X, dim, stride):
    W1, H1, C1 = X.shape
    W2 = round((W1 - dim + 1) / stride + 1)
    H2 = round((H1 - dim + 1) / stride + 1)
    output_fmap = np.zeros((W2, H2, C1), dtype=float)
    for k in range(C1):
        x = 0
        for w in np.arange(0, W1 - dim - 1, stride):
            y = 0
            for h in np.arange(0, H1 - dim - 1, stride):
                output_fmap[x, y, k] = np.max(X[w:w + dim, h:h + dim, k])
                output_fmap[x, y, k] = relu(output_fmap[x, y, k])
                y += 1
            x += 1
    return np.asarray(output_fmap, dtype=uint8)


def softmax(X):
    X = np.float64(X)
    out = np.exp(X)
    return out / np.sum(out)


IMAGE_PATH = "1.jpg"
img = cv2.imread(IMAGE_PATH)

convolution_stride = 1
pooling_stride = 2
pooling_window_dim = 2
initial_shift = np.zeros(5, dtype=float)
kernel = np.random.rand(3, 3)
filters = np.array([[kernel, kernel, kernel],
                    [kernel, kernel, kernel],
                    [kernel, kernel, kernel],
                    [kernel, kernel, kernel],
                    [kernel, kernel, kernel]])

conv_layer = conv(img, filters, initial_shift, convolution_stride)
normalization_layer = normalize(conv_layer)
polling_layer = max_pooling(normalization_layer, pooling_window_dim, pooling_stride)
probs = softmax(polling_layer)

print(probs)
