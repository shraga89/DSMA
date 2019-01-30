import DataHandler as DH
import Config as conf
import BaseAdaptor as BA
import AdaptAndEvaluate as AnE
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import datetime
import time
import random

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
    fm = BA.bulid_fm()

    for epoch in train:
        X_seq = dh.conf_dict_seq[epoch]
        X_mat = dh.conf_dict_mat[epoch]
        y_seq = dh.realConf_dict[epoch]
        y_mat = dh.realConf_dict_mat[epoch]
        y_single = dh.fullMat_dict[epoch][E]

        # print(X_mat[0].reshape(X_mat.shape[1:3]))
        fm.fit(X_seq.reshape(X_seq.shape[1]), y_seq.reshape(y_seq.shape[1]))

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

        res_adapt, count_adapt = AnE.reg_adapt_fm(np.array(dh.inv_trans[epoch]), 'FM', X_seq, y_seq, fm,
                                                   matM, matN, res_adapt, count_adapt)
    i += 1

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M')
folder = './results/' + st + '_' + dataset + '_' + E
if not os.path.exists(folder):
    os.makedirs(folder)
res_adapt.to_csv(folder + '/adapt.csv')
res_adapt_eval.to_csv(folder + '/adapt_eval.csv')
res_eval.to_csv(folder + '/eval.csv')