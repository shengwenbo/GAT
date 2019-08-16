import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process
import scipy.sparse as sp

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'cora'

# training params
batch_size = 50
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
max_neighs = 500 # maximum number of neighbors
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [1, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

adj = adj.todense()

# features = features[np.newaxis]
# adj = adj[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

ids_real = process.devide_graph(features, adj, len(n_heads), max_neighs)

features = np.concatenate([features, np.zeros([1, ft_size])], 0)
adj = np.concatenate([adj, np.zeros([nb_nodes, 1])], 1)
adj = np.concatenate([adj, np.zeros([1, nb_nodes + 1])], 0)

# adj = adj + sp.eye(nb_nodes)

train_ids = [id for id in range(nb_nodes) if train_mask[id] == True]
val_ids = [id for id in range(nb_nodes) if val_mask[id] == True]
test_ids = [id for id in range(nb_nodes) if test_mask[id] == True]

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(nb_nodes + 1, ft_size))
        adj_in = tf.placeholder(dtype=tf.float32, shape=(nb_nodes + 1, nb_nodes + 1))
        ids_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, max_neighs))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())

    logits = model.inference([ids_in, ftr_in, adj_in], nb_classes, max_neighs, is_train,
                                attn_drop, ffd_drop,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            np.random.shuffle(train_ids)
            np.random.shuffle(val_ids)

            tr_step = 0
            tr_size = len(train_ids)

            while tr_step * batch_size < tr_size:
                if (tr_step + 1)*batch_size < tr_size:
                    ids = train_ids[tr_step*batch_size: (tr_step+1)*batch_size]
                else:
                    ids = train_ids[tr_step*batch_size: ] + train_ids[0: (tr_step+1)*batch_size - tr_size]
                _, loss_value_tr, acc_tr, logits_tr = sess.run([train_op, loss, accuracy, logits],
                    feed_dict={
                        ids_in: ids_real[ids],
                        ftr_in: features,
                        adj_in: adj,
                        lbl_in: y_train[ids],
                        msk_in: train_mask[ids],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr

                tr_step += 1

            vl_step = 0
            vl_size = len(val_ids)//5

            while vl_step * batch_size < vl_size:
                ids = val_ids[vl_step*batch_size: (vl_step+1)*batch_size]
                loss_value_vl, acc_vl = sess.run([loss, accuracy],
                    feed_dict={
                        ids_in: ids_real[ids],
                        ftr_in: features,
                        adj_in: adj,
                        lbl_in: y_val[ids],
                        msk_in: val_mask[ids],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = len(test_ids)
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        test_logs = []
        test_lbls = []

        while ts_step * batch_size < ts_size:
            ids = test_ids[ts_step*batch_size:(ts_step+1)*batch_size]
            loss_value_ts, acc_ts, log = sess.run([loss, accuracy, log_resh],
                feed_dict={
                    ids_in: ids_real[ids],
                    ftr_in: features,
                    adj_in: adj,
                    lbl_in: y_test[ids],
                    msk_in: test_mask[ids],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
            test_logs.append(log)
            test_lbls.append(y_test[0])


        # y = y_train + y_val + y_test
        # lbl_all = np.argmax(y[0], axis=-1).tolist()
        # for logs in test_logs:
        #     pred = np.argmax(logs, axis=-1).tolist()
        #
        #     for p, r, i in zip(pred, lbl_all, range(len(pred))):
        #         if p != r:
        #             print("ID: {}".format(i))
        #             print("Pred: {}, real: {}.".format(p, r))
        #
        #             ftr = features[0, i, :]
        #             bias = biases[0, i, :]
        #             nbs = [j for j in range(len(pred)) if bias[j] > -1]
        #             ftr_nb = features[0, nbs, :]
        #
        #             print("Neighbors: {}".format(nbs))
        #             # print("Feature: {}".format(ftr))
        #             print("Neighbor labels: {}".format([lbl_all[nb] for nb in nbs]))
        #             # print("Neighbor Features: {}".format(ftr_nb))
        #             print()

        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()
