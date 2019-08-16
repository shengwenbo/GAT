import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            hid_units, n_heads, activation=tf.nn.elu, residual=False):
        ids, features, adj = inputs
        features = tf.nn.embedding_lookup(features, ids)
        adjs = GAT.embedding_lookup_2d(adj, ids)
        bias_mat = GAT.get_bias(adjs)
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head1(features, adj=adjs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head1(h_1, adj=adjs, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head1(h_1, adj=adjs, bias_mat=bias_mat,
                out_sz=nb_classes, activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        logits = logits[:, 0, :]
        # logits = tf.nn.dropout(logits, 1 - ffd_drop)
        # logits = layers.conv1d(logits, nb_classes, 1, use_bias=False)
        # logits = tf.reduce_mean(logits, 1)

        return logits

    def embedding_lookup_2d(adj, ids):
        bs = ids.shape[0]
        adjs = []
        for i in range(bs):
            a = tf.nn.embedding_lookup(adj, ids[i])
            a = tf.transpose(a, [1, 0])
            a = tf.nn.embedding_lookup(a, ids[i])
            adjs.append(tf.expand_dims(a, 0))
        return tf.concat(adjs, 0)

    def get_bias(adjs):
        return -1e9 * (1.0 - adjs)