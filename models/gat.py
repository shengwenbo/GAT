import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, input_dim, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, split_mode, split_parts, activation=tf.nn.elu, residual=False):
        attns = []
        # inputs = tf.layers.dense(inputs,input_dim, use_bias=False, kernel_initializer=tf.random_normal_initializer, trainable=False)
        if split_mode == "train_share":
            sp_wei = tf.get_variable("sp_wei_{}".format("in"), [split_parts[0], hid_units[0]])
        elif split_mode == "random_const":
            sp_wei = np.random.random(size=(split_parts[0], hid_units[0]))
            sp_wei = np.exp(sp_wei) / np.sum(sp_wei, axis=0)
            sp_wei = tf.constant(sp_wei, dtype=tf.float32)
        else:
            sp_wei = None
        for i in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                split_mode=split_mode, split_parts=split_parts[0], sp_wei=sp_wei,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, name="attn_{}_{}".format("in",i))[0])
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            if split_mode == "train_share":
                sp_wei = tf.get_variable("sp_wei_{}".format(i), [split_parts[i], hid_units[i]])
            elif split_mode == "random_const":
                sp_wei = np.random.random(size=(split_parts[i], hid_units[i]))
                sp_wei = np.exp(sp_wei) / np.sum(sp_wei, axis=0)
                sp_wei = tf.constant(sp_wei, dtype=tf.float32)
            else:
                sp_wei = None
            h_old = h_1
            attns = []
            for j in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                split_mode=split_mode, split_parts=split_parts[i], sp_wei=sp_wei,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual, name="attn_{}_{}".format(i,j))[0])
            h_1 = tf.concat(attns, axis=-1)
        out = []
        if split_mode == "train_share":
            sp_wei = tf.get_variable("sp_wei_{}".format("out"), [split_parts[-1], nb_classes])
        elif split_mode == "random_const":
            sp_wei = np.random.random(size=(split_parts[-1], nb_classes))
            sp_wei = np.exp(sp_wei) / np.sum(sp_wei, axis=0)
            sp_wei = tf.constant(sp_wei, dtype=tf.float32)
        else:
            sp_wei = None
        attn=[]
        for i in range(n_heads[-1]):
            attn.append(layers.attn_head(h_1, bias_mat=bias_mat,
                split_mode=split_mode, split_parts=split_parts[-1], sp_wei=sp_wei,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False, name="attn_{}_{}".format("out", i)))
        for a in attn:
            out.append(a[0])
        key_vecs = attn[0][1]
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits, key_vecs


class GAT_old(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, input_dim, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        # inputs = tf.layers.dense(inputs, input_dim, use_bias=False, kernel_initializer=tf.random_normal_initializer, trainable=True)
        for i in range(n_heads[0]):
            attns.append(layers.attn_head_old(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False)[0])
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for j in range(n_heads[i]):
                attns.append(layers.attn_head_old(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual)[0])
            h_1 = tf.concat(attns, axis=-1)
        out = []
        attn = []
        for i in range(n_heads[-1]):
            attn.append(layers.attn_head_old(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        for a in attn:
            out.append(a[0])
        key_vecs = attn[0][1]
        logits = tf.add_n(out) / n_heads[-1]

        return logits, key_vecs