import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, activation, bias_mat=None, split_mode="random_const", split_parts=2, sp_wei=None, in_drop=0.0, coef_drop=0.0, residual=False, name="attn"):
    n_nodes = seq.shape[1]
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts_sp = []
        for _ in range(split_parts):
            seq_fts_sp.append(tf.expand_dims(tf.layers.conv1d(seq, out_sz, 1, use_bias=False), 0))
        seq_fts_sp = tf.concat(seq_fts_sp, 0)
        seq_fts = tf.reduce_mean(seq_fts_sp, 0)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) # [bs, n, 1]
        f_2 = tf.reshape(seq_fts_sp, [-1, n_nodes, out_sz]) # [sp*bs, n, d]
        f_2 = tf.layers.conv1d(f_2, 1, 1) # [sp*bs, n, 1]
        f_2 = tf.reshape(f_2, [split_parts, -1, n_nodes, 1]) # [sp, bs, n, 1]
        logits = tf.expand_dims(f_1, 0) + tf.transpose(f_2, [0, 1, 3, 2]) # [sp, bs, n, n]
        in_coefs = tf.nn.softmax(tf.nn.leaky_relu(logits), axis=0) # [sp, bs, n, n]
        logits = in_coefs * logits
        if bias_mat is not None:
            coefs = tf.nn.softmax(tf.nn.leaky_relu(tf.reduce_sum(logits, axis=0)) + bias_mat) # [bs, n, n]
        else:
            coefs = tf.nn.softmax(tf.nn.leaky_relu(tf.reduce_sum(logits, axis=0)))  # [bs, n, n]
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

def attn_head_sep(seq, ids, out_sz, activation, sparse=False, split_parts=2, attn_size=16, sp_wei=None, in_drop=0.0, coef_drop=0.0, residual=False, name="attn"):
    if sparse:
        n_nodes = ids.dense_shape[1]
    else:
        n_nodes = ids.shape[1]
    with tf.name_scope('my_attn'):

        if in_drop != 0.0:
            seq = tf.layers.dropout(seq, 1.0 - in_drop)

        seq_fts_sp = []
        seq = tf.expand_dims(seq, 1) # [ns, 1, fd]
        for _ in range(split_parts):
            sf = tf.layers.dense(seq, out_sz, use_bias=False) # [ns, 1, d]
            sf = tf.nn.embedding_lookup(tf.reshape(sf, [-1, out_sz]), ids)
            seq_fts_sp.append(sf)
        seq_fts_sp = tf.stack(seq_fts_sp, 2) # [bs, n, sp, d]
        seq_fts = tf.reduce_mean(seq_fts_sp, 2) # [bs, n, d]
        cnt_fts = seq_fts[:, 0:1, :] # [bs, 1, d]

        # simplest self-attention possible
        logits = attn(tf.expand_dims(cnt_fts, 1), seq_fts_sp, attn_size) # [bs, n, 1, sp]
        in_coefs = tf.nn.softmax(tf.nn.leaky_relu(logits), axis=-1) # [bs, n, 1, sp]

        # if coef_drop != 0.0:
        #     in_coefs = tf.nn.dropout(in_coefs, 1.0 - coef_drop)
        # if in_drop != 0.0:
        #     seq_fts_sp = tf.nn.dropout(seq_fts_sp, 1.0 - in_drop)

        new_seq_fts = tf.matmul(in_coefs, seq_fts_sp) # [bs, n, 1, d]
        new_seq_fts = tf.reshape(new_seq_fts, [-1, n_nodes, out_sz]) # [bs, n, d]
        logits_new = attn(cnt_fts, new_seq_fts, attn_size) # [bs, 1, n]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits_new), axis=-1) # [bs, 1, n]

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            new_seq_fts = tf.nn.dropout(new_seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, new_seq_fts) # [bs, 1, d]
        ret = tf.reshape(vals, [-1, out_sz])
        # ret = tf.contrib.layers.bias_add(ret)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, split_parts=4, in_drop=0.0, coef_drop=0.0,
                     residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts_sp = []
        for _ in range(split_parts):
            seq_fts_sp.append(tf.expand_dims(tf.layers.conv1d(seq, out_sz, 1, use_bias=False), 0))
        seq_fts_sp = tf.concat(seq_fts_sp, 0)
        seq_fts = tf.reduce_mean(seq_fts_sp, 0)

        logits = []
        for sp in range(split_parts):
            f_1 = tf.layers.conv1d(seq_fts, 1, 1)
            f_2 = tf.layers.conv1d(seq_fts_sp[sp, :, :], 1, 1)  # [bs, n, 1]

            f_1 = tf.reshape(f_1, (nb_nodes, 1))
            f_2 = tf.reshape(f_2, (nb_nodes, 1))

            f_1 = adj_mat * f_1
            f_2 = adj_mat * tf.transpose(f_2, [1, 0])

            logits.append(tf.sparse_reshape(tf.sparse_add(f_1, f_2), [1, nb_nodes, nb_nodes]))
        logits = tf.sparse_concat(0, logits)

        logits_tr = tf.sparse_transpose(logits, [1, 2, 0])
        lrelu = tf.SparseTensor(indices=logits_tr.indices,
                                values=tf.nn.leaky_relu(logits_tr.values),
                                dense_shape=logits_tr.dense_shape)
        in_coefs = tf.sparse_softmax(lrelu)
        in_coefs = tf.sparse_transpose(in_coefs, [2, 0, 1])

        logits = logits * logits
        logits = tf.sparse_reduce_sum_sparse(logits, axis=0)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)
        coefs = tf.sparse_reshape(coefs, [1, nb_nodes, nb_nodes])
        coefs = coefs*in_coefs

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        vals = []
        for sp in range(split_parts):
            coefs_i = tf.sparse_reshape(coefs[sp, :, :], [nb_nodes, nb_nodes])
            seq_fts_sp_i = tf.squeeze(seq_fts_sp[sp, :, :])
            vals.append(tf.sparse_reshape(tf.sparse_tensor_dense_matmul(coefs_i, seq_fts_sp_i), [1, nb_nodes, out_sz]))
        vals = tf.sparse_concat(0, vals)
        vals = tf.sparse_reduce_sum_sparse(vals, 0)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head_old(seq, out_sz, adj_mat, activation, nb_nodes, split_parts=4, in_drop=0.0, coef_drop=0.0, residual=False):
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

def attn(f1, f2, attn_size):

    f1 = tf.layers.dense(f1, attn_size)
    f2 = tf.layers.dense(f2, attn_size)
    logits = f1 * f2
    logits = tf.reduce_sum(logits, -1)
    logits = tf.expand_dims(logits, -2)

    return logits