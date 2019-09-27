import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle as pkl

origin_path = r"D:\Users\Administrator\Desktop\data\Pumbed\Pubmed-Diabetes\data"
out_path = "./data"

tr_size = 60
vl_size = 500
ts_size = 1000

labels = pd.DataFrame()
id2idx = {}
idx2id = {}

with open(os.path.join(origin_path, "Pubmed-Diabetes.NODE.paper.tab")) as fin:
    fin.readline()
    fin.readline()
    idx = 0

    word_list = set([])
    label_list = set([])
    lines = fin.readlines()
    for line in lines:
        line = line.strip()
        splits = line.split()
        lb_dis = splits[1]
        lb = lb_dis.split("=")[1]
        ft_dis = splits[2:-1]

        for fd in ft_dis:
            ft_name, _ = fd.split("=")
            word_list.add(ft_name)

        label_list.add(lb)

    word_list = list(word_list)
    list.sort(word_list)
    print(word_list)
    w2v = {}
    for i,w in zip(range(len(word_list)), word_list):
        w2v[w] = i

    label_list = list(label_list)
    list.sort(label_list)
    print(label_list)
    l2v = {}
    for i,l in zip(range(len(label_list)), label_list):
        l2v[l] = i

    features = []
    labels = []
    for line in lines:
        line = line.strip()
        splits = line.split()
        id = splits[0]
        lb_dis = splits[1]
        ft_dis = splits[2:-1]

        lb = lb_dis.split("=")[1]
        id2idx[id] = idx
        idx2id[idx] = id

        ft = np.zeros(len(word_list))
        for fd in ft_dis:
            ft_name, ft_value = fd.split("=")
            ft[w2v[ft_name]] = 1.0
        features.append(ft)

        label = np.zeros((len(label_list)))
        label[l2v[lb]] = 1.0
        labels.append(label)
        idx += 1

    features = np.stack(features, 0)
    labels = np.stack(labels, 0)

graph = {}
with open(os.path.join(origin_path, "Pubmed-Diabetes.DIRECTED.cites.tab")) as fin:
    fin.readline()
    fin.readline()
    for line in fin.readlines():
        line = line.strip()
        id, src, _, tgt = line.split()
        src = src.split(":")[1]
        tgt = tgt.split(":")[1]

        if src in id2idx.keys():
            src = id2idx[src]
        else:
            print(src)

        if tgt in id2idx.keys():
            tgt = id2idx[tgt]
        else:
            print(tgt)

        if src in graph.keys():
            graph[src].append(tgt)
        else:
            graph[src] = [tgt]
        if tgt in graph.keys():
            graph[tgt].append(src)
        else:
            graph[tgt] = [src]

edges = list(graph.items())
graph = {}
list.sort(edges, key=lambda x: x[0])
for s, t in edges:
    list.sort(t)
    graph[s] = t

n_nodes = features.shape[0]
test_index = list(range(n_nodes-ts_size, n_nodes))
test_index = [str(i) for i in test_index]
assert len(test_index) == ts_size
x = features[0:tr_size, :]
y = labels[0:tr_size, :]
all_x = features[0:n_nodes-ts_size, :]
all_y = labels[0:n_nodes-ts_size, :]
tx = features[n_nodes-ts_size:n_nodes, :]
ty = labels[n_nodes-ts_size:n_nodes, :]

for obj,name in zip([x, y, all_x, all_y, tx, ty, graph],
                    ["x", "y", "allx", "ally", "tx", "ty", "graph"]):
    path = os.path.join(out_path, "ind.pubmed.{}".format(name))
    pkl.dump(obj, open(path, "wb"), protocol=-1)

with open(os.path.join(out_path, "ind.pubmed.test.index"), "w", encoding="utf-8") as fout:
    fout.write("\n".join(test_index))

print("finish!")
