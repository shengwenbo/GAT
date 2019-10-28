# Author: Narine Kokhlikyan <narine@slice.com>
# License: BSD

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

import sys
import os
import shutil
import random
from utils import process

import pickle as pkl

COLOR={
    0:"b",
    1:"g",
    2:"r",
    3:"c",
    4:"m",
    5:"y",
    6:"k"
}

dataset = "cora"
train_size = 140
n_classes = 7
# key_vecs_path = "./analyze/train.key_vecs"
# key_vecs_path = "./analyze/train.hid_vecs"
key_vecs_path = "./analyze/origin.key_vecs"
# key_vecs_path = "./analyze/origin.log_vecs"

n_components = 2

center = 1851
distant = 2

with open("data/ind.{}.{}".format(dataset, "graph"), 'rb') as f:
    if sys.version_info > (3, 0):
        graph = pkl.load(f, encoding='latin1')
    else:
        graph = pkl.load(f)
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, ally = process.load_data(dataset, train_size, class_balanced=True)

ids = set([center])
nei = graph[center]
for d in range(distant):
    ids_ = ids.copy()
    for i in ids_:
        ids = ids | set(graph[i])
ids = list(ids)
list.sort(ids)

srcs = []
tgts = []
for s in graph.keys():
    for t in graph[s]:
        if s >= t :
            continue
        srcs.append(s)
        tgts.append(t)

X = pkl.load(open(key_vecs_path, "rb"))
if len(X.shape) == 2:
    X = X[1:, :]
    X = np.expand_dims(X, 0)
if X.shape[0] > 1:
    X = X[1:, :, :]
    X = np.transpose(X, [1, 0, 2])
    X_sum = np.sum(X, axis=0)
    X_sum = np.expand_dims(X_sum, 0)
    X = np.append(X, X_sum, axis=0)

y = np.argmax(ally, axis=-1)

n_dims = X.shape[2]
n_splits = X.shape[0]
n_samples = X.shape[1]

classes = {}
for i in range(n_classes):
    classes[i] = [j for j in range(n_samples) if y[j] == i and j in ids]

for split in range(n_splits):
    plt.figure(figsize=(15, 8))

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=50)
    X_ = tsne.fit_transform(X[split, :, :])
    t1 = time()
    print("circles in %.2g sec" % (t1 - t0))

    plt.title("%s/Split %d" % (key_vecs_path, split))

    for src, tgt in zip(srcs, tgts):
        if src not in ids or tgt not in ids:
            continue
        plt.plot([X_[src, 0], X_[tgt, 0]], [X_[src, 1], X_[tgt, 1]], c="gray", lw=0.9, alpha=0.5)

    plt.scatter(X_[center, 0], X_[center, 1], marker="*", c=COLOR[y[center]], s=500)
    for i in nei:
        plt.scatter(X_[i, 0], X_[i, 1], marker="^", c=COLOR[y[i]], s=300)

    for c in range(n_classes):
        plt.scatter(X_[classes[c], 0], X_[classes[c], 1], c=COLOR[c], edgecolors="w")

    for i in ids:
        plt.annotate(str(i), xy=(X_[i, 0], X_[i, 1]), xytext=(X_[i, 0] + 0.1, X_[i, 1] + 0.1))

    plt.axis('tight')

plt.show()