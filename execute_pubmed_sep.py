import time
import numpy as np
import tensorflow as tf
import sys
import os
import shutil
import scipy.sparse as sp

from models import SEP_GAT
from utils import process
import random
dataset = 'pubmed'
# dataset = 'cora'

# training params
nb_epochs = 20000
patience = 10
lr = 0.01  # learning rate
l2_coef = 0.002  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
# n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = SEP_GAT
log_every = 10
if dataset == "citeseer":
    max_nei = 50
elif dataset == "cora":
    max_nei = 128
elif dataset == "pubmed":
    max_nei = 64

sparse = False

if __name__ == "__main__":
    checkpt_file = 'pre_trained/cora/mod_cora.ckpt'
    split_mode = sys.argv[1]# ["random", "random_const", "train", "train_no_softmax", "train_share"]
    split_parts = sys.argv[2].split(",")
    split_parts = [int(a) for a in split_parts]
    n_heads = sys.argv[3].split(",")
    n_heads = [int(a) for a in n_heads]
    in_drop = float(sys.argv[4])
    coef_drop = float(sys.argv[5])
    train_size = int(sys.argv[6])
    out_dir = sys.argv[7]
    log_dir = sys.argv[8]

    seed = os.path.basename(out_dir)
    seed = int(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    turn = int(out_dir.strip().split("/")[-1])
    tf.random_normal_initializer.seed = turn

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

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset, train_size, class_balanced=True)
    features, spars = process.preprocess_features(features)
    # features = features.todense()

    adj = np.array(adj.todense())

    features = np.concatenate((np.zeros([1, features.shape[1]]), features), axis=0)
    adj = np.concatenate((np.zeros([1, adj.shape[1]]), adj), axis=0)
    adj = np.concatenate((np.zeros([adj.shape[0], 1]), adj), axis=1)
    y_train = np.concatenate((np.zeros([1, y_train.shape[1]], dtype=np.float32), y_train), axis=0)
    y_val = np.concatenate((np.zeros([1, y_val.shape[1]], dtype=np.float32), y_val), axis=0)
    y_test = np.concatenate((np.zeros([1, y_test.shape[1]], dtype=np.float32), y_test), axis=0)
    train_mask = np.concatenate((np.zeros(1, dtype=np.bool), train_mask), axis=0)
    val_mask = np.concatenate((np.zeros(1, dtype=np.bool), val_mask), axis=0)
    test_mask = np.concatenate((np.zeros(1, dtype=np.bool), test_mask), axis=0)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    batch_size = nb_nodes

    # biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)
    adj_list = process.adj_to_list(adj, max_nei=max_nei)

    if sparse:
        adj_list = sp.csr_matrix(adj_list, dtype=np.int64).tocoo()
        adj_list = (
            np.vstack((adj_list.col, adj_list.row)).transpose(),
            adj_list.data,
            adj_list.shape
        )

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(nb_nodes, ft_size))
            if sparse:
                ids_in = tf.sparse_placeholder(dtype=tf.int32)
            else:
                ids_in = tf.placeholder(dtype=tf.int32, shape=(None, max_nei))
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(None, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(None))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

            logits = model.inference(ids_in, ftr_in, nb_classes, nb_nodes, is_train,
                                    attn_drop, ffd_drop,
                                    split_mode=split_mode, split_parts=split_parts,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity,
                                    sparse=sparse)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("acc", accuracy)
        merged = tf.summary.merge_all()

        train_log_dir = "{}/train/{}".format(log_dir, "_".join(sys.argv[1:7]))
        val_log_dir = "{}/val/{}".format(log_dir, "_".join(sys.argv[1:7]))
        for dir in [train_log_dir, val_log_dir]:
            if os.path.exists(dir):
                shutil.rmtree(dir)
        train_log_writer = tf.summary.FileWriter(train_log_dir, tf.get_default_graph())
        val_log_writer = tf.summary.FileWriter(val_log_dir)

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
                tr_step = 0
                tr_size = features.shape[0]

                while tr_step * batch_size < tr_size:
                    _, loss_value_tr, acc_tr, summary_tr = sess.run([train_op, loss, accuracy, merged],
                        feed_dict={
                            ftr_in: features,
                            ids_in: adj_list[tr_step*batch_size:(tr_step+1)*batch_size],
                            lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                            msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                            is_train: True,
                            attn_drop: in_drop, ffd_drop: coef_drop})
                    if epoch%log_every == 0:
                        train_log_writer.add_summary(summary_tr, epoch)
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                vl_step = 0
                vl_size = features.shape[0]

                while vl_step * batch_size < vl_size:
                    loss_value_vl, acc_vl, summary_vl = sess.run([loss, accuracy, merged],
                        feed_dict={
                            ftr_in: features,
                            ids_in: adj_list[vl_step*batch_size:(vl_step+1)*batch_size],
                            lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                            msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                            is_train: False,
                            attn_drop: 0.0, ffd_drop: 0.0})
                    if epoch % log_every == 0:
                        val_log_writer.add_summary(summary_vl, epoch)
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                print('Training #%d/%d: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                        (epoch, tr_step, train_loss_avg/tr_step, train_acc_avg/tr_step,
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

            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            test_logs = []
            test_lbls = []

            while ts_step * batch_size < ts_size:
                loss_value_ts, acc_ts, log = sess.run([loss, accuracy, log_resh],
                    feed_dict={
                        ftr_in: features,
                        ids_in: adj_list[ts_step*batch_size:(ts_step+1)*batch_size],
                        lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                        msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1
                test_logs.append(log)
                test_lbls.append(y_test[0])

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            y = y_train + y_val + y_test
            lbl_all = np.argmax(y, axis=-1).tolist()
            with open("{}/{}_{}.txt".format(out_dir, "_".join(sys.argv[1:7]), "test_out"), "w") as fout:
                for logs in test_logs:
                    pred = np.argmax(logs, axis=-1).tolist()

                    for p, r, i in zip(pred, lbl_all, range(len(pred))):
                        if p != r:
                            fout.write("ID: {}\n".format(i))
                            fout.write("Pred: {}, real: {}.\n".format(p, r))

                            ftr = features[i, :]
                            nbs = [adj_list[i, j] for j in range(max_nei) if adj_list[i, j] > 0]
                            ftr_nb = features[nbs, :]

                            fout.write("Neighbors: {}\n".format(nbs))
                            # print("Feature: {}".format(ftr))
                            fout.write("Neighbor labels: {}\n\n".format([lbl_all[nb] for nb in nbs]))
                            # print("Neighbor Features: {}".format(ftr_nb))

            with open("{}/{}_{}.txt".format(out_dir, "_".join(sys.argv[1:7]), ts_acc/ts_step), "w") as fout:
                fout.write("{} {} {} {} {}".format("Hid units:", hid_units, "; Num heads:", n_heads, "\n"))
                fout.write("{} {} {} {} {}".format("Split mode:", split_mode, "; Split parts:", split_parts, "\n"))
                fout.write("{} {} {} {} {}".format("In drop:", in_drop, "; Coef drop:", coef_drop, "\n"))
                fout.write("{} {} {}".format("Total epochs:", epoch, "\n"))
                fout.write("{} {} {} {} {}".format('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step, "\n"))

            sess.close()
