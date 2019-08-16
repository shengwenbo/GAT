import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        center_ft = seq_fts[:, 0:1, :]

        # simplest self-attention possible
        f_0 = tf.layers.conv1d(center_ft, 1, 1)
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1]) + f_0
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

def attn_head1(seq, out_sz, adj, bias_mat, activation, classes=4, in_drop=0.0, coef_drop=0.0, residual=False):
    nb_nodes = seq.shape[1]
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        class_fts = tf.layers.conv1d(seq, classes, 1, use_bias=False)
        class_logs = tf.nn.softmax(tf.nn.leaky_relu(class_fts), -1)
        class_logs = tf.transpose(class_logs, [2, 0, 1])

        class_logs = tf.expand_dims(class_logs, -2)
        adj = tf.expand_dims(adj, 0)
        class_adj = class_logs * adj
        nei_fts = tf.tile(seq_fts, [classes, 1, 1])
        nei_fts = tf.reshape(nei_fts, [classes, -1, nb_nodes, out_sz])
        nei_fts = tf.matmul(class_adj, nei_fts)
        nei_fts = tf.transpose(nei_fts, [1, 2, 0, 3])

        seq_dense = seq_fts
        nei_dense = nei_fts
        seq_dense = tf.expand_dims(seq_dense, 2)
        wei = seq_dense * nei_dense
        wei = tf.reduce_sum(wei, -1)
        wei = tf.nn.softmax(tf.nn.leaky_relu(wei), -1)
        wei = tf.expand_dims(wei, -1)

        if coef_drop is not None:
            wei = tf.nn.dropout(wei, 1-coef_drop)
        if in_drop is not None:
            nei_fts = tf.nn.dropout(nei_fts, 1-in_drop)

        nei_values = wei * nei_fts
        nei_values = tf.reduce_sum(nei_values, 2)

        ret = seq_fts + nei_values

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
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

