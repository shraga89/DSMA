import DataHandler as DH
import DeepAdaptor as DA
import BaseAdaptor as BA
import DeepEvaluator as DE
import ModelVisualizer as MV
import AdaptAndEvaluate as AnE
import FeatureBasedEvaluator as FBE
import MultiTaskNet as MTN
import Config as conf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import datetime
import time
import random
import multiprocessing as mp
import os
import keras
from keras import backend as K
import tensorflow as tf


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=28))
tf.Session(config=tf.ConfigProto(log_device_placement=True))
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 28})
# config = tf.ConfigProto(device_count={'GPU': 1}, intra_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
# config.allow_soft_placement = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def adapt_worker(dh, X_seq, X_mat, y_seq, matM, matN, ir, svd, bpr,
                 gru_model_adapt, cnn_model_adapt, dnn_model_adapt, crnn_model_adapt, multi_model):
    global res_adapt, count_adapt
    res_adapt, count_adapt = AnE.reg_adapt_ir(np.array(dh.inv_trans[epoch]), 'IR', X_seq, y_seq, ir,
                                              res_adapt, count_adapt)
    res_adapt, count_adapt = AnE.reg_adapt_svd(np.array(dh.inv_trans[epoch]), 'SVDpp', X_seq, y_seq, svd,
                                               matM, matN, res_adapt, count_adapt)
    res_adapt, count_adapt = AnE.reg_adapt_bpr(np.array(dh.inv_trans[epoch]), 'BPR', X_mat, X_seq, y_seq, bpr,
                                               res_adapt, count_adapt)
    res_adapt, count_adapt = AnE.only_deep_adapt(np.array(dh.inv_trans[epoch]), 'CRNN', X_seq, y_seq,
                                                 crnn_model_adapt, res_adapt, count_adapt)


def eval_worker(dh, X_feat, X_seq, y_single, gru_model_eval, cnn_model_eval, dnn_model_eval, crnn_model_eval,
                multi_model):
    global res_eval, count_eval
    res_eval, count_eval = AnE.only_reg_evaluate(np.array(dh.inv_trans[epoch]), 'FeatureBased_', X_feat, y_single,
                                                 FBE.classifiers, res_eval, count_eval)
    cos_val = dh.fullMat_dict[epoch]['cos']
    res_eval, count_eval = AnE.only_deep_evaluate(np.array(dh.inv_trans[epoch]), 'CRNN', X_seq, y_single,
                                                  crnn_model_eval, res_eval, count_eval, cos_val)

print(K.tensorflow_backend._get_available_gpus())
E = conf.E
dataset = conf.dataset
dh = DH.DataHandler(conf.datafile, None, conf.syn)
dh.build_eval(False)
dh.build_feat_dataset()
# print(dh.conf_dict)
kfold = KFold(5, True, 1)
keys = np.array(list(dh.conf_dict.keys()))[:]
# keys = np.array(random.sample(range(len(list(dh.conf_dict.keys()))), 100))
res_adapt = pd.DataFrame(columns=['instance', 'type', 'k', 'old_p', 'old_r', 'old_f', 'new_p', 'new_r', 'new_f'])
res_adapt_eval = pd.DataFrame(columns=['instance', 'type', 'k', 'old_p', 'old_r',
                                       'old_f', 'new_p', 'new_r', 'new_f', 'real_e', 'pred_e'])
res_eval = pd.DataFrame(columns=['instance', 'type', 'k', 'real_p', 'pred_p', 'sum', 'cos'])
i = 1
count_adapt, count_eval, count_adapt_eval = 0, 0, 0
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
print(st)
for train, test in kfold.split(keys):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
    print("Starting fold " + str(i) + ' ' + str(st))
    K.get_session().close()
    K.set_session(tf.Session())
    K.get_session().run(tf.global_variables_initializer())
    K.clear_session()
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
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
        print("Starting test fold " + str(i) + ' ' + str(st))
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

        adapt_worker(dh, X_seq, X_mat, y_seq, matM, matN, ir, svd, bpr, None, None, None, crnn_model_adapt, None)

        eval_worker(dh, X_feat, X_seq, y_single, None, None, None, crnn_model_eval, None)

        # CRNN_CRNN
        res_adapt_eval, count_adapt_eval = AnE.deep_adapt_and_evaluate(np.array(dh.inv_trans[epoch]),
                                                                       'CRNN_CRNN', crnn_model_adapt,
                                                                       False, crnn_model_eval, False, X_seq,
                                                                       X_mat, y_seq, y_single, res_adapt_eval,
                                                                       count_adapt_eval)
    i += 1
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
folder = './results/' + st + '_' + dataset + '_' + E
if not os.path.exists(folder):
    os.makedirs(folder)
res_adapt.to_csv(folder + '/adapt.csv')
res_adapt_eval.to_csv(folder + '/adapt_eval.csv')
res_eval.to_csv(folder + '/eval.csv')
