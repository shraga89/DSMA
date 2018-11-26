import DataHandler as DH
import DeepAdaptor as DA
import BaseAdaptor as BA
import DeepEvaluator as DE
import ModelVisualizer as MV
import AdaptAndEvaluate as AnE
import FeatureBasedEvaluator as FBE
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf
import datetime
import time
import os
import random
import multiprocessing as mp
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=28))


def model_worker_adapt(model, X, y, e, b, v):
    mat = X.reshape(X.shape[1:3])
    exMat = y.reshape(y.shape[1:3])
    X_seq = mat.reshape(1, mat.shape[0]*mat.shape[1], 1)
    y_seq = mat.reshape(1, exMat.shape[0] * exMat.shape[1], 1)
    model.fit(X_seq, y_seq, epochs=e, batch_size=b, verbose=v)
    X_seq = mat.T.reshape(1, mat.shape[0]*mat.shape[1], 1)
    y_seq = mat.reshape(1, exMat.shape[0] * exMat.shape[1], 1)
    model.fit(X_seq, y_seq, epochs=e, batch_size=b, verbose=v)
    for _ in range(4):
        i = random.randint(0, mat.shape[0]-1)
        j = random.randint(0, mat.shape[0]-1)
        mat_i = mat[i]
        mat[i] = mat[j]
        mat[j] = mat_i
        exMat_i = exMat[i]
        exMat[i] = exMat[j]
        exMat[j] = exMat_i
        X_seq = mat.reshape(1, mat.shape[0] * mat.shape[1], 1)
        y_seq = mat.reshape(1, exMat.shape[0] * exMat.shape[1], 1)
        model.fit(X_seq, y_seq, epochs=e, batch_size=b, verbose=v)
        i = random.randint(0, mat.shape[1]-1)
        j = random.randint(0, mat.shape[1]-1)
        mat_i = mat[:, i]
        mat[:, i] = mat[:, j]
        mat[:, j] = mat_i
        exMat_i = exMat[:, i]
        exMat[:, i] = exMat[:, j]
        exMat[:, j] = exMat_i
        X_seq = mat.reshape(1, mat.shape[0] * mat.shape[1], 1)
        y_seq = mat.reshape(1, exMat.shape[0] * exMat.shape[1], 1)
        model.fit(X_seq, y_seq, epochs=e, batch_size=b, verbose=v)
    return model


def model_worker_eval(model, X, y, e, b, v):
    mat = X.reshape(X.shape[1:3])
    X_seq = mat.reshape(1, mat.shape[0]*mat.shape[1], 1)
    model.fit(X_seq, y_seq, epochs=e, batch_size=b, verbose=v)
    X_seq = mat.T.reshape(1, mat.shape[0]*mat.shape[1], 1)
    model.fit(X_seq, y, epochs=e, batch_size=b, verbose=v)
    for _ in range(4):
        i = random.randint(0, mat.shape[0]-1)
        j = random.randint(0, mat.shape[0]-1)
        mat_i = mat[i]
        mat[i] = mat[j]
        mat[j] = mat_i
        exMat_i = exMat[i]
        exMat[i] = exMat[j]
        exMat[j] = exMat_i
        X_seq = mat.reshape(1, mat.shape[0] * mat.shape[1], 1)
        model.fit(X_seq, y, epochs=e, batch_size=b, verbose=v)
        i = random.randint(0, mat.shape[1]-1)
        j = random.randint(0, mat.shape[1]-1)
        mat_i = mat[:, i]
        mat[:, i] = mat[:, j]
        mat[:, j] = mat_i
        exMat_i = exMat[:, i]
        exMat[:, i] = exMat[:, j]
        exMat[:, j] = exMat_i
        X_seq = mat.reshape(1, mat.shape[0] * mat.shape[1], 1)
        model.fit(X_seq, y, epochs=e, batch_size=b, verbose=v)
    return model


def adapt_worker(dh, X_seq, X_mat, y_seq, matM, matN, ir, svd, bpr,
                 gru_model_adapt, cnn_model_adapt, dnn_model_adapt, crnn_model_adapt):
    global res_adapt, count_adapt
    res_adapt, count_adapt = AnE.reg_adapt_ir(np.array(dh.inv_trans[epoch]), 'IR', X_seq, y_seq, ir,
                                              res_adapt, count_adapt)
    res_adapt, count_adapt = AnE.reg_adapt_svd(np.array(dh.inv_trans[epoch]), 'SVDpp', X_seq, y_seq, svd,
                                               matM, matN, res_adapt, count_adapt)
    res_adapt, count_adapt = AnE.reg_adapt_bpr(np.array(dh.inv_trans[epoch]), 'BPR', X_mat, X_seq, y_seq, bpr,
                                               res_adapt, count_adapt)

    res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'GRU', X_seq, y_seq,
                                                 gru_model_adapt, res_adapt, count_adapt)
    res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'CNN', X_seq, y_seq,
                                                 cnn_model_adapt, res_adapt, count_adapt)
    res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'DNN', X_seq, y_seq,
                                                 dnn_model_adapt, res_adapt, count_adapt)
    res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'CRNN', X_seq, y_seq,
                                                 crnn_model_adapt, res_adapt, count_adapt)


def eval_worker(dh, X_feat, X_seq, y_single, gru_model_eval, cnn_model_eval, dnn_model_eval, crnn_model_eval):
    global res_eval, count_eval
    res_eval, count_eval = AnE.only_reg_evaluate(np.array(dh.inv_trans[epoch]), 'FeatureBased_', X_feat, y_single,
                                                 FBE.classifiers, res_eval, count_eval)

    res_eval, count_eval = AnE.only_deep_evaluate(np.array(dh.inv_trans[epoch]), 'GRU', X_seq, y_single,
                                                  gru_model_eval, res_eval, count_eval)
    res_eval, count_eval = AnE.only_deep_evaluate(np.array(dh.inv_trans[epoch]), 'CNN', X_seq, y_single,
                                                  cnn_model_eval, res_eval, count_eval)
    res_eval, count_eval = AnE.only_deep_evaluate(np.array(dh.inv_trans[epoch]), 'DNN', X_seq, y_single,
                                                  dnn_model_eval, res_eval, count_eval)
    res_eval, count_eval = AnE.only_deep_evaluate(np.array(dh.inv_trans[epoch]), 'CRNN', X_seq, y_single,
                                                  crnn_model_eval, res_eval, count_eval)


E = 'f'
dh = DH.DataHandler('../VectorsWFsmall.csv', '../_matrix.csv', True)
dh.build_eval(False)
dh.build_feat_dataset()
kfold = KFold(5, True, 1)
keys = np.array(list(dh.conf_dict.keys()))[:5]
# keys = np.array(random.sample(range(len(list(dh.conf_dict.keys()))), 100))
res_adapt = pd.DataFrame(columns=['instance', 'type', 'k', 'old_p', 'old_r', 'old_f', 'new_p', 'new_r', 'new_f'])
res_adapt_eval = pd.DataFrame(columns=['instance', 'type', 'k', 'old_p', 'old_r',
                                       'old_f', 'new_p', 'new_r', 'new_f', 'real_e', 'pred_e'])
res_eval = pd.DataFrame(columns=['instance', 'type', 'k', 'real_p', 'pred_p'])
i = 1
count_adapt, count_eval, count_adapt_eval = 0, 0, 0
for train, test in kfold.split(keys):
    print("Starting fold " + str(i))
    K.get_session().close()
    K.set_session(tf.Session())
    K.get_session().run(tf.global_variables_initializer())
    K.clear_session()
    gru_model_adapt = DA.build_gru(32)
    gru_model_eval = DE.build_gru(32)
    cnn_model_adapt = DA.build_cnn(64)
    cnn_model_eval = DE.build_cnn(16)
    cnn_2d_model_eval = DE.build_cnn_2d(16)
    dnn_model_adapt = DA.build_dnn(64)
    dnn_model_eval = DE.build_dnn(64)
    crnn_model_adapt = DA.build_cnn_gru(16)
    crnn_model_eval = DE.build_cnn_gru(16)

    ir = BA.build_ir()
    svd = BA.build_svdpp()
    bpr = BA.build_bpr()
    # pool = mp.Pool()
    for epoch in train:
        # DEEP
        X_seq = dh.conf_dict_seq[epoch]
        X_mat = dh.conf_dict_mat[epoch]
        y_seq = dh.realConf_dict[epoch]
        y_mat = dh.realConf_dict_mat[epoch]
        y_single = dh.fullMat_dict[epoch][E]
        gru_model_adapt = model_worker_adapt(gru_model_adapt, X_mat, y_mat, 1, 1, 2)
        cnn_model_adapt = model_worker_adapt(cnn_model_adapt, X_mat, y_mat, 1, 1, 2)
        dnn_model_adapt = model_worker_adapt(dnn_model_adapt, X_mat, y_mat, 1, 1, 2)
        crnn_model_adapt = model_worker_adapt(crnn_model_adapt, X_mat, y_mat, 1, 1, 2)
        gru_model_eval = model_worker_eval(gru_model_eval, X_mat, y_single, 1, 1, 2)
        cnn_model_eval = model_worker_eval(cnn_model_eval, X_mat, y_single, 1, 1, 2)
        dnn_model_eval = model_worker_eval(dnn_model_eval, X_mat, y_single, 1, 1, 2)
        crnn_model_eval = model_worker_eval(crnn_model_eval, X_mat, y_single, 1, 1, 2)
        cnn_2d_model_eval.fit(X_mat, y_single, epochs=1, batch_size=1, verbose=2)

        # REG ADAPT
        ir.fit(X_seq.reshape(X_seq.shape[1]), y_seq.reshape(y_seq.shape[1]))

        # REG EVAL
        X_feat = dh.feat_dict[epoch]
        for clf in FBE.classifiers:
            clf[1].fit(X_feat, y_single)

    mv_adapt = MV.ModelVisualizer(gru_model_adapt, dh)
    mv_adapt.visualize_gru(1, False)
    mv_eval = MV.ModelVisualizer(gru_model_eval, dh)
    mv_eval.visualize_gru(1, True)
    mv_eval_cnn = MV.ModelVisualizer(cnn_2d_model_eval, dh)
    mv_eval_cnn.visualize_cnn(1, True)
    mv_adapt = MV.ModelVisualizer(gru_model_adapt, dh)
    mv_adapt.visualize_gru(0, False)
    mv_eval = MV.ModelVisualizer(gru_model_eval, dh)
    mv_eval.visualize_gru(0, True)
    mv_eval_cnn = MV.ModelVisualizer(cnn_2d_model_eval, dh)
    mv_eval_cnn.visualize_cnn(0, True)
    DH.data_saver(gru_model_adapt, "./models/gru_adapt_no_attention_model_fold_" + str(i))
    DH.data_saver(cnn_model_adapt, "./models/cnn_adapt_no_attention_model_fold_" + str(i))
    DH.data_saver(dnn_model_adapt, "./models/dnn_adapt_no_attention_model_fold_" + str(i))
    DH.data_saver(crnn_model_adapt, "./models/crnn_adapt_no_attention_model_fold_" + str(i))
    DH.data_saver(gru_model_eval, "./models/gru_eval_model_fold_" + str(i))
    DH.data_saver(cnn_model_eval, "./models/cnn_eval_model_fold_" + str(i))
    DH.data_saver(dnn_model_eval, "./models/dnn_eval_model_fold_" + str(i))
    DH.data_saver(crnn_model_eval, "./models/crnn_eval_model_fold_" + str(i))
    # pool.close()
    # pool = mp.Pool()
    for epoch in test:
        if 'exact' in dh.inv_trans[epoch]:
            print('skipping Exact Match')
            continue
        X_seq = dh.conf_dict_seq[epoch]
        X_mat = dh.conf_dict_mat[epoch]
        X_feat = dh.feat_dict[epoch]
        y_seq = dh.realConf_dict[epoch]
        y_mat = dh.realConf_dict_mat[epoch]
        y_seq = np.array(y_seq.reshape(len(y_seq[0]), 1))
        y_single = dh.fullMat_dict[epoch][E]
        matN = dh.matN[epoch]
        matM = dh.matM[epoch]

        adapt_worker(dh, X_seq, X_mat, y_seq, matM, matN, ir, svd, bpr,
                     gru_model_adapt, cnn_model_adapt, dnn_model_adapt, crnn_model_adapt)

        eval_worker(dh, X_feat, X_seq, y_single, gru_model_eval, cnn_model_eval, dnn_model_eval, crnn_model_eval)


            # GRU_GRU
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'GRU_GRU', gru_model_adapt,
                                                                           False, gru_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single,
                                                                           res_adapt_eval, count_adapt_eval)

            # CNN_CNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'CNN_CNN', cnn_model_adapt,
                                                                           False, cnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single,
                                                                           res_adapt_eval, count_adapt_eval)

            # DNN_DNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'DNN_DNN', dnn_model_adapt,
                                                                           False, dnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # CRNN_CRNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'CRNN_CRNN', crnn_model_adapt,
                                                                           False, crnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # GRU_CNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'GRU_CNN', gru_model_adapt,
                                                                           False, cnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # GRU_DNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'GRU_DNN', gru_model_adapt,
                                                                           False, dnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # GRU_CRNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'GRU_CRNN', gru_model_adapt,
                                                                           False, crnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # CNN_GRU
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'CNN_GRU', cnn_model_adapt,
                                                                           False, gru_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # CNN_DNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'CNN_DNN', cnn_model_adapt,
                                                                           False, dnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # CNN_CRNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'CNN_CRNN', cnn_model_adapt,
                                                                           False, crnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # DNN_GRU
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'DNN_GRU', dnn_model_adapt,
                                                                           False, gru_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # DNN_CNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'DNN_CNN', dnn_model_adapt,
                                                                           False, cnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # DNN_CRNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'DNN_CRNN', dnn_model_adapt,
                                                                           False, crnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # CRNN_GRU
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'CRNN_GRU', crnn_model_adapt,
                                                                           False, gru_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # CRNN_CNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'CRNN_CNN', crnn_model_adapt,
                                                                           False, cnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

            # CRNN_DNN
            res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'DNN_CRNN', crnn_model_adapt,
                                                                           False, dnn_model_eval, False, X_seq,
                                                                           X_mat, y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

        res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_reg_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'GRU_FEAT', gru_model_adapt,
                                                                           FBE.classifiers, X_seq, matN, matM, X_feat,
                                                                           y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

        res_adapt_eval, count_adapt_eval = AnE.ir_adapt_and_deep_evaluate(np.array(dh.inv_trans[epoch]),
                                                                          'IR_GRU', ir,
                                                                          gru_model_eval, X_seq,
                                                                          y_seq, y_single, res_adapt_eval,
                                                                          count_adapt_eval)

        res_adapt_eval, count_adapt_eval = AnE.svd_adapt_and_deep_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'SVD_GRU', svd,
                                                                           gru_model_eval, X_seq, matN, matM,
                                                                           y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

        res_adapt_eval, count_adapt_eval = AnE.bpr_adapt_and_deep_evaluate(np.array(dh.inv_trans[epoch]),
                                                                           'BPR_GRU', bpr,
                                                                           gru_model_eval, X_seq, matN, matM,
                                                                           y_seq, y_single, res_adapt_eval,
                                                                           count_adapt_eval)

        res_adapt_eval, count_adapt_eval = AnE.ir_adapt_and_reg_evaluate(np.array(dh.inv_trans[epoch]),
                                                                         'IR_FEAT', ir,
                                                                         FBE.classifiers, X_seq, matN, matM, X_feat,
                                                                         y_seq, y_single, res_adapt_eval,
                                                                         count_adapt_eval)

        res_adapt_eval, count_adapt_eval = AnE.svd_adapt_and_reg_evaluate(np.array(dh.inv_trans[epoch]),
                                                                          'SVD_FEAT', svd,
                                                                          FBE.classifiers, X_seq, matN, matM, X_feat,
                                                                          y_seq, y_single, res_adapt_eval,
                                                                          count_adapt_eval)

        res_adapt_eval, count_adapt_eval = AnE.bpr_adapt_and_reg_evaluate(np.array(dh.inv_trans[epoch]),
                                                                          'BPR_FEAT', bpr,
                                                                          FBE.classifiers, X_seq, matN, matM, X_feat,
                                                                          y_seq, y_single, res_adapt_eval,
                                                                          count_adapt_eval)

    i += 1
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
folder = './results/' + st
if not os.path.exists(folder):
    os.makedirs(folder)
res_adapt.to_csv(folder + '/adapt.csv')
res_adapt_eval.to_csv(folder + '/adapt_eval.csv')
res_eval.to_csv(folder + '/eval.csv')
# res_adapt = pd.read_csv(folder + '/adapt.csv')
# res_adapt_eval = pd.read_csv(folder + '/adapt_eval.csv')
# res_eval = pd.read_csv(folder + '/eval.csv')
# AnE.summerize(res_adapt, folder + '/adapt_sum.csv')
# AnE.summerize(res_adapt_eval, folder + '/adapt_eval_sum.csv')
# AnE.summerize(res_eval, folder + '/eval_sum.csv')
