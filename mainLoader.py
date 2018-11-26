import DataHandler as DH
import BaseAdaptor as BA
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
import multiprocessing as mp

E = 'f'
dh = DH.DataHandler('../VectorsWF.csv', '../_matrix.csv')
dh.build_eval(False)
kfold = KFold(5, True, 1)
# keys = np.array(list(dh.conf_dict.keys()))[:100]
keys = np.array(random.sample(range(len(list(dh.conf_dict.keys()))), 100))
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
    gru_model_adapt = DH.data_loader("./models/11_11_2018_08_42/gru_adapt_no_attention_model_fold_" + str(i))
    gru_model_eval = DH.data_loader("./models/11_11_2018_08_42/gru_eval_model_fold_" + str(i))
    cnn_model_adapt = DH.data_loader("./models/11_11_2018_08_42/cnn_adapt_no_attention_model_fold_" + str(i))
    cnn_model_eval = DH.data_loader("./models/11_11_2018_08_42/cnn_eval_model_fold_" + str(i))
    dnn_model_adapt = DH.data_loader("./models/11_11_2018_08_42/dnn_adapt_no_attention_model_fold_" + str(i))
    dnn_model_eval = DH.data_loader("./models/11_11_2018_08_42/dnn_eval_model_fold_" + str(i))
    crnn_model_adapt = DH.data_loader("./models/11_11_2018_08_42/crnn_adapt_no_attention_model_fold_" + str(i))
    crnn_model_eval = DH.data_loader("./models/11_11_2018_08_42/crnn_eval_model_fold_" + str(i))

    ir = BA.build_ir()
    svd = BA.build_svdpp()
    bpr = BA.build_bpr()
    for epoch in train:
        # DEEP
        X_seq = dh.conf_dict_seq[epoch]
        X_mat = dh.conf_dict_mat[epoch]
        y_seq = dh.realConf_dict[epoch]
        y_single = dh.fullMat_dict[epoch][E]

        # REG ADAPT
        ir.fit(X_seq.reshape(X_seq.shape[1]), y_seq.reshape(y_seq.shape[1]))

        # REG EVAL
        X_feat = dh.feat_dict[epoch]
        for clf in FBE.classifiers:
            clf[1].fit(X_feat, y_single)

    for epoch in test:
        print('epoch number: ' + str(epoch))
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

        res_adapt, count_adapt = AnE.reg_adapt_ir(np.array(dh.inv_trans[epoch]), 'IR', X_seq, y_seq, ir,
                                                  res_adapt, count_adapt)
        res_adapt, count_adapt = AnE.reg_adapt_svd(np.array(dh.inv_trans[epoch]), 'SVDpp', X_seq, y_seq, svd,
                                                   dh.matM[epoch], dh.matN[epoch], res_adapt, count_adapt)
        # res_adapt, count_adapt = AnE.reg_adapt_bpr(np.array(dh.inv_trans[epoch]), 'BPR', X_mat, X_seq, y_seq, bpr,
        #                                            res_adapt, count_adapt)

        res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'GRU', X_seq, y_seq,
                                                     gru_model_adapt, res_adapt, count_adapt)
        res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'CNN', X_seq, y_seq,
                                                     cnn_model_adapt, res_adapt, count_adapt)
        res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'DNN', X_seq, y_seq,
                                                     dnn_model_adapt, res_adapt, count_adapt)
        res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'CRNN', X_seq, y_seq,
                                                     crnn_model_adapt, res_adapt, count_adapt)

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

    i += 1
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
folder = './results/' + st
if not os.path.exists(folder):
    os.makedirs(folder)
res_adapt.to_csv(folder + '/adapt.csv')
res_adapt_eval.to_csv(folder + '/adapt_eval.csv')
res_eval.to_csv(folder + '/eval.csv')
