import time
import numpy as np
import tensorflow as tf

import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)

    return adj, features, labels, train_mask, val_mask, test_mask

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


dataset = 'reddit'
old_prefix = 'D:/data/reddit_10-0/reddit'

adj, features, y, train_mask, val_mask, test_mask = load_data(dataset)

G = {
    "directed": False,
    "graph": {},
    "nodes": [],
    "links": [],
    "multigraph": False
}

id_map = {}
class_map = {}

nodes = []

nodes_cnt = features.shape[0]

idx = 0
for id in range(nodes_cnt):
    if train_mask[id]:
        test = False
        val = False
        label = y[id]
    elif test_mask[id]:
        test = True
        val = False
        label = y[id]
    elif val_mask[id]:
        test = False
        val = True
        label = y[id]
    else:
        test = True
        val = False
        label = y[id]

    for i in range(len(label)):
        if label[i] > 0:
            label = i
            break

    print(id, label, test, val)

    id_map[id] = idx
    class_map[id] = label

    nodes.append({
        "id": id,
        "test": test,
        "val": val,
    })

    idx += 1

G["nodes"] = nodes

adj = adj.tocoo()

for row,col in zip(adj.row, adj.col):

    if row not in id_map.keys() or col not in id_map.keys():
        continue

    link = {
        "source": id_map[row],
        "target": id_map[col]
    }
    G["links"].append(link)

features = features.todense()[np.array(list(id_map.keys()), dtype=int)]

G = json.load(open(old_prefix + "-G.json", 'r', encoding="utf-8"))
features = np.load(old_prefix + "-feats.npy")
id_map = json.load(open(old_prefix + "-id_map.json", "r", encoding="utf-8"))
class_map = json.load(open(old_prefix + "-class_map.json", "r", encoding="utf-8"))