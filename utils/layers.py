import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, split_mode="random_const", split_parts=2, sp_wei=None, in_drop=0.0, coef_drop=0.0, residual=False, name="attn"):
    n_nodes = seq.shape[1]
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        if split_mode == "random_const":
            sp_wei = sp_wei
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, sp, d]
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, 1, sp, d]
        elif split_mode == "random":
            sp_wei = tf.random_normal([split_parts, out_sz])
            sp_wei = tf.nn.softmax(sp_wei, axis=0)  # [sp, d]
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, sp, d]
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, 1, sp, d]
        elif split_mode == "train":
            sp_wei = tf.get_variable("sp_wei_" + name, [split_parts, out_sz])
            # sp_wei = tf.nn.softmax(sp_wei, axis=0)  # [sp, d]
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, sp, d]
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, 1, sp, d]
        elif split_mode == "train_share":
            sp_wei = sp_wei
            # sp_wei = tf.nn.softmax(sp_wei, axis=0)  # [sp, d]
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, sp, d]
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, 1, sp, d]
        elif split_mode == "train_no_softmax":
            sp_wei = tf.get_variable("sp_wei_" + name, [split_parts, out_sz])
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, sp, d]
            sp_wei = tf.expand_dims(sp_wei, 0)  # [1, 1, sp, d]

        seq_fts_sp = tf.expand_dims(seq_fts, 2) # [bs, n, 1, d]
        seq_fts_sp = seq_fts_sp * sp_wei # [bs, n, sp, d]
        seq_fts_sp = tf.transpose(seq_fts_sp, [2, 0, 1, 3]) # [sp, bs, n, d]

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) # [bs, n, 1]
        f_2 = tf.reshape(seq_fts_sp, [-1, n_nodes, out_sz]) # [sp*bs, n, d]
        f_2 = tf.layers.conv1d(f_2, 1, 1) # [sp*bs, n, 1]
        f_2 = tf.reshape(f_2, [split_parts, -1, n_nodes, 1]) # [sp, bs, n, 1]
        logits = tf.expand_dims(f_1, 0) + tf.transpose(f_2, [0, 1, 3, 2]) # [sp, bs, n, n]
        in_coefs = tf.nn.softmax(tf.nn.leaky_relu(logits), axis=0) # [sp, bs, n, n]
        logits = logits * logits
        coefs = tf.nn.softmax(tf.nn.leaky_relu(tf.reduce_sum(logits, axis=0)) + bias_mat) # [bs, n, n]
        coefs = tf.expand_dims(coefs, 0) # [1, bs, n, n]
        coefs = coefs * in_coefs # [sp, bs, n, n]

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts_sp = tf.nn.dropout(seq_fts_sp, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts_sp) # [sp, bs, n, b]
        vals = tf.reduce_sum(vals, 0) # [bs, n, b]
        # vals = []
        # for i in range(split_parts):
        #     vals.append(tf.matmul(coefs[i], seq_fts_sp[i]))
        # vals = tf.add_n(vals)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

def attn_head_old(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, split_parts=4, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) # [bs, n, 1]
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) # [bs, m, sp, 1]
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

