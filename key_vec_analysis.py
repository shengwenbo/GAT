import time
import numpy as np
import tensorflow as tf
import sys
import os
import shutil
import pickle as pkl

from utils import process

dataset = 'cora'
train_size = 140
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset, train_size, class_balanced=True)

features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()

features = features[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

key_vecs = pkl.load("origin.key_vecs")
