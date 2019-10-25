import time
import numpy as np
import tensorflow as tf
import sys
import os
import shutil
import pickle as pkl
import random

from models import GAT, GAT_old
from utils import process

# dataset = 'citeseer'
dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 200
input_dim = 512
patience = 10
lr = 0.01  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
# n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT
log_every = 10

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

    biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

        if split_mode == "origin":
            model = GAT_old
            logits, key_vecs = model.inference(ftr_in, nb_classes, nb_nodes, input_dim, is_train,
                                    attn_drop, ffd_drop,
                                    bias_mat=bias_in,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity)
        else:
            logits, key_vecs = model.inference(ftr_in, nb_classes, nb_nodes, input_dim, is_train,
                                    attn_drop, ffd_drop,
                                    split_mode=split_mode, split_parts=split_parts,
                                    bias_mat=bias_in,
                                    hid_units=hid_units, n_heads=n_heads,
                                    residual=residual, activation=nonlinearity)
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
                            ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                            bias_in: biases[tr_step*batch_size:(tr_step+1)*batch_size],
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
                            ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                            bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
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
                loss_value_ts, acc_ts, key_vecs_data, log = sess.run([loss, accuracy, key_vecs, log_resh],
                    feed_dict={
                        ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                        bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                        lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                        msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1
                test_logs.append(log)
                test_lbls.append(y_test[0])

            pkl.dump(key_vecs_data, open("{}.key_vecs".format(split_mode), "wb"))

            stdout_old = sys.stdout
            sys.stdout = open("{}/{}_{}.false".format(out_dir, "_".join(sys.argv[1:7]), "test_out"), "w")
            y = y_train + y_val + y_test
            lbl_all = np.argmax(y[0], axis=-1).tolist()
            err_total = 0
            for logs in test_logs:
                pred = np.argmax(logs, axis=-1).tolist()
                # pred_p = [float(logs[i, p]) for i, p in zip(range(len(pred), pred))]

                for p, r, i in zip(pred, lbl_all, range(len(pred))):
                    if p != r:
                        err_total += 1
                        print("NO. {}".format(err_total))
                        print("ID: {}".format(i))
                        print("Pred: {}, real: {}.".format(p, r))

                        ftr = features[0, i, :]
                        bias = biases[0, i, :]
                        nbs = [j for j in range(len(pred)) if bias[j] > -1 and j != i]
                        nbs2 = []
                        lbl2 = []
                        lbl2_p = []
                        for nb in nbs:
                            bias = biases[0, nb, :]
                            nb2 = [j for j in range(len(pred)) if bias[j] > -1 and j != nb]
                            l2 = [lbl_all[nb] for nb in nb2]
                            l2_p = [pred[nb] for nb in nb2]
                            nbs2.append(nb2)
                            lbl2_p.append(l2_p)
                            lbl2.append(l2)
                        ftr_nb = features[0, nbs, :]

                        print("Neighbors: {}".format(nbs))
                        # print("Feature: {}".format(ftr))
                        lbl_p = [pred[nb] for nb in nbs]
                        print("Neighbor labels pred: {}".format(lbl_p))
                        # print("Neighbor labels conf: {}".format([float(logs[n, l]) for n,l in zip(nbs, lbl_p)]))
                        print("Neighbor labels: {}".format([lbl_all[nb] for nb in nbs]))
                        print("2 layer neighbors: {}".format(nbs2))
                        print("2 layer neighbor labels pred: {}".format(lbl2_p))
                        print("2 layer neighbor labels: {}".format(lbl2))
                        # print("Neighbor Features: {}".format(ftr_nb))
                        print()
            sys.stdout = stdout_old

            sys.stdout = open("{}/{}_{}.true".format(out_dir, "_".join(sys.argv[1:7]), "test_out"), "w")
            y = y_train + y_val + y_test
            lbl_all = np.argmax(y[0], axis=-1).tolist()
            err_total = 0
            for logs in test_logs:
                pred = np.argmax(logs, axis=-1).tolist()
                # pred_p = [float(logs[i, p]) for i, p in zip(range(len(pred), pred))]

                for p, r, i in zip(pred, lbl_all, range(len(pred))):
                    if p == r:
                        err_total += 1
                        print("NO. {}".format(err_total))
                        print("ID: {}".format(i))
                        print("Pred: {}, real: {}.".format(p, r))

                        ftr = features[0, i, :]
                        bias = biases[0, i, :]
                        nbs = [j for j in range(len(pred)) if bias[j] > -1 and j != i]
                        nbs2 = []
                        lbl2 = []
                        lbl2_p = []
                        for nb in nbs:
                            bias = biases[0, nb, :]
                            nb2 = [j for j in range(len(pred)) if bias[j] > -1 and j != nb]
                            l2 = [lbl_all[nb] for nb in nb2]
                            l2_p = [pred[nb] for nb in nb2]
                            nbs2.append(nb2)
                            lbl2_p.append(l2_p)
                            lbl2.append(l2)
                        ftr_nb = features[0, nbs, :]

                        print("Neighbors: {}".format(nbs))
                        # print("Feature: {}".format(ftr))
                        lbl_p = [pred[nb] for nb in nbs]
                        print("Neighbor labels pred: {}".format(lbl_p))
                        # print("Neighbor labels conf: {}".format([float(logs[n, l]) for n,l in zip(nbs, lbl_p)]))
                        print("Neighbor labels: {}".format([lbl_all[nb] for nb in nbs]))
                        print("2 layer neighbors: {}".format(nbs2))
                        print("2 layer neighbor labels pred: {}".format(lbl2_p))
                        print("2 layer neighbor labels: {}".format(lbl2))
                        # print("Neighbor Features: {}".format(ftr_nb))
                        print()
            sys.stdout = stdout_old

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            with open("{}/{}_{}.txt".format(out_dir, "_".join(sys.argv[1:7]), ts_acc/ts_step), "w") as fout:
                fout.write("{} {} {} {} {}".format("Hid units:", hid_units, "; Num heads:", n_heads, "\n"))
                fout.write("{} {} {} {} {}".format("Split mode:", split_mode, "; Split parts:", split_parts, "\n"))
                fout.write("{} {} {} {} {}".format("In drop:", in_drop, "; Coef drop:", coef_drop, "\n"))
                fout.write("{} {} {}".format("Total epochs:", epoch, "\n"))
                fout.write("{} {} {} {} {}".format('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step, "\n"))

            sess.close()
