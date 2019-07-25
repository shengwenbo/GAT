import os
import sys
import pickle as pkl
import numpy as np
import random
import scipy.sparse as sp

PATH = "./citeseer_data/"
LBLS = ["Agents", "AI", "DB", "IR", "ML", "HCI"]
TRAIN_SIZE = 120
TEST_SIZE = 1000

def load_citeseer():
    allx, ally, x, y, tx, ty, test_index, id2idx = load_content()
    graph = load_graph()

    idx_graph = {}
    for source, target in graph:
        if source not in id2idx.keys() or target not in id2idx.keys():
            print(source, target)
            continue
        idx_s = id2idx[source]
        idx_t = id2idx[target]
        if idx_s not in idx_graph.keys():
            idx_graph[idx_s] = []
        idx_graph[idx_s].append(idx_t)

    objects = [allx, ally, idx_graph, tx, ty, x, y]
    i = 0
    for name in ["allx", "ally", "graph", "tx", "ty", "x", "y"]:
        with open(os.path.join(PATH, "ind.citeseer.{}".format(name)), 'wb') as f:
            if sys.version_info > (3, 0):
                pkl.dump(objects[i], f, protocol=-1)
            else:
                pkl.dump(objects[i], f)
        i += 1
    with open(os.path.join(PATH, "ind.citeseer.test.index"), 'w', encoding="utf-8") as f:
        [f.write(str(i)+"\n") for i in test_index]


def load_content():
    with open(os.path.join(PATH, "citeseer.content"), "r", encoding="utf-8") as fin:
        id2idx = {}
        i = 0
        ids = []
        ftrs = []
        lbls = []
        for line in fin.readlines():
            parts = line.strip().split()
            id, ftr, lbl = parts[0], parts[1:-1], parts[-1]
            id2idx[id] = i
            ids.append(id)
            ftrs.append(ftr)
            lbls.append(lbl2array(lbl))
            i += 1

        total = i

        train_ids = random.sample(ids, k=TRAIN_SIZE)
        test_ids = random.sample(list(set(ids) - set(train_ids)), k=TEST_SIZE)
        other_ids = list(set(ids) - set(train_ids) - set(test_ids))

        ids = train_ids + other_ids + test_ids
        idxs = [id2idx[id] for id in ids]

        ftrs = [ftrs[idx] for idx in idxs]
        lbls = [lbls[idx] for idx in idxs]
        for (id, i) in zip(ids, list(range(len(ids)))):
            id2idx[id] = i

        allx = list2csr(ftrs[0 : total - TEST_SIZE])
        ally = np.array(lbls[0 : total - TEST_SIZE])
        x = list2csr(ftrs[0 : TRAIN_SIZE])
        y = np.array(lbls[0 : TRAIN_SIZE])
        tx = list2csr(ftrs[total - TEST_SIZE : ])
        ty = np.array(lbls[total - TEST_SIZE : ])
        test_index = list(range(total - TEST_SIZE, total))

        return allx, ally, x, y, tx, ty, test_index, id2idx

def load_graph():
    with open(os.path.join(PATH, "citeseer.cites"), "r", encoding="utf-8") as fin:
        graph = []
        for line in fin.readlines():
            source, target = line.strip().split()
            graph.append((source, target))
        return graph

def lbl2array(lbl):
    idx = LBLS.index(lbl)
    a = np.zeros(len(LBLS))
    a[idx] = 1.0
    return a

def list2csr(x):
    x = np.array(x, dtype=np.float32)
    return sp.csr_matrix(x)


load_citeseer()