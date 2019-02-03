import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import to_categorical
from keras.models import model_from_json
import random
import multiprocessing as mp
import FeatureBasedEvaluator as FBE


def data_saver(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    print("Saved " + model_name + " to disk")


def data_loader(model_name):
    json_file = open(model_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")
    print(str(model_name) + " loaded model from disk")
    return loaded_model


def random_1():
    return random.betavariate(alpha=0.6406, beta=0.2040)


def random_0():
    return random.betavariate(alpha=2.6452, beta=16.3139)


class DataHandler:

    def __init__(self, reg_flie, feat_file, syn):
        self.reg_flie = reg_flie
        self.reg_df = None
        self.create_reg()
        # self.feat_file = feat_file
        # self.feat_df = None
        # self.create_feat()
        self.conf_dict = {}
        self.conf_dict_mat = {}
        self.conf_dict_seq = {}
        self.realConf_dict = {}
        self.realConf_dict_mat = {}
        self.trans = {}
        self.matN = {}
        self.matM = {}
        self.build_dataset_reg()
        # if syn:
        #     self.transform_2_syntatic()
        self.orig_mats_mat = {}
        self.orig_mats_seq = {}
        self.fullMat_dict = {}
        self.inv_trans = {v: k for k, v in self.trans.items()}
        self.feat_dict = {}
        self.orig_results = pd.DataFrame(columns=['instance', 'P', 'R', 'F', 'COS'])

    def create_reg(self):
        # self.reg_df = pd.read_csv(self.reg_flie, low_memory=False, error_bad_lines = False)
        # self.reg_df['pair'] = self.reg_df['candName'] + '<->' + self.reg_df['targName']
        mylist = []
        for chunk in pd.read_csv(self.reg_flie, low_memory=False, chunksize=10 ** 6):
            mylist.append(chunk)
        self.reg_df = pd.concat(mylist, axis=0)
        self.reg_df['pair'] = self.reg_df['candName'] + '<->' + self.reg_df['targName']
        del mylist


    # def create_feat(self):
    #     self.feat_df = pd.read_csv(self.feat_file, low_memory=False)

    def build_dataset_reg(self):
        i = 0
        for name, group in self.reg_df[['instance', 'candName', 'targName', 'conf', 'realConf']].groupby(
                by=["instance"]):
            clean_name = name.replace(",", " ").replace("  ", " ")
            if i not in self.conf_dict:
                self.trans[clean_name] = i
                i += 1
                self.conf_dict[self.trans[clean_name]] = np.array([])
                self.matN[self.trans[clean_name]] = group['targName'].value_counts()[-1]
                self.matM[self.trans[clean_name]] = group['candName'].value_counts()[-1]
            if i not in self.realConf_dict:
                self.realConf_dict[self.trans[clean_name]] = np.array([])
            self.conf_dict[self.trans[clean_name]] = np.append(self.conf_dict[self.trans[clean_name]],
                                                               np.array(group['conf']))
            self.realConf_dict[self.trans[clean_name]] = np.append(self.realConf_dict[self.trans[clean_name]],
                                                                   np.array(group['realConf']))

    def build_eval(self, classes):
        if classes:
            for k in self.conf_dict:
                self.fullMat_dict[k] = to_categorical(
                    np.around(cosine_similarity(self.conf_dict[k].reshape(1, -1),
                                                self.realConf_dict[k].reshape(1, -1)) * 10), num_classes=11)
                self.conf_dict_mat[k] = self.conf_dict[k].reshape(1, self.matN[k], self.matM[k], 1)
                self.conf_dict_seq[k] = self.conf_dict[k].reshape(1, len(self.conf_dict[k]), 1)
        else:
            for k in self.conf_dict:
                self.fullMat_dict[k] = {}
                self.fullMat_dict[k]['p'], self.fullMat_dict[k]['r'], self.fullMat_dict[k][
                    'f'] = precision_recall_fscore_support(
                    np.ceil(np.array(self.conf_dict[k].reshape(len(self.conf_dict[k]), 1))),
                    np.array(self.realConf_dict[k].reshape(len(self.realConf_dict[k]), 1)))[:3]
                for e in self.fullMat_dict[k]:
                    self.fullMat_dict[k][e] = np.array(self.fullMat_dict[k][e]).reshape(1, 1)
                self.fullMat_dict[k]['cos'] = cosine_similarity(self.conf_dict[k].reshape(1, -1),
                                                                self.realConf_dict[k].reshape(1, -1))
                if 'exactMatch' not in self.inv_trans[k]:
                    res_row = np.concatenate((self.inv_trans[k], self.fullMat_dict[k]['p'], self.fullMat_dict[k]['r'],
                                              self.fullMat_dict[k]['f'], self.fullMat_dict[k]['cos']), axis=None)
                    self.orig_results.loc[k] = res_row
                self.conf_dict_mat[k] = self.conf_dict[k].reshape(1, self.matN[k], self.matM[k], 1)
                self.conf_dict_seq[k] = self.conf_dict[k].reshape(1, len(self.conf_dict[k]), 1)
                self.orig_mats_mat[k] = self.conf_dict[k].reshape(self.matN[k], self.matM[k])
                self.orig_mats_seq[k] = self.conf_dict[k].reshape(len(self.conf_dict[k]))
                if np.isnan(np.min(self.conf_dict[k])):
                    print("**********")
                self.realConf_dict[k] = np.array(self.realConf_dict[k].reshape(1, len(self.realConf_dict[k]), 1))
                self.realConf_dict_mat[k] = np.array(self.realConf_dict[k].reshape(1, self.matN[k], self.matM[k], 1))

    # def build_feat_dataset(self):
    #     for name, group in self.feat_df.groupby(by=["instance"]):
    #         if self.trans[name] not in self.feat_dict:
    #             self.feat_dict[self.trans[name]] = np.array(group.drop(['instance', 'F'], axis=1))

    def build_feat_dataset(self):
        for k in self.conf_dict_mat:
            self.feat_dict[k] = FBE.extractPreds(self.conf_dict_mat[k].reshape(self.matN[k], self.matM[k]))

    def transform_2_syntatic(self):
        for k in self.realConf_dict:
            self.conf_dict[k] = np.array([])
            for i in self.realConf_dict[k]:
                if i == 1.0:
                    self.conf_dict[k] = np.append(self.conf_dict[k], np.array(random_1()))
                else:
                    self.conf_dict[k] = np.append(self.conf_dict[k], np.array(random_0()))
            self.conf_dict[k] = np.where(self.conf_dict[k] > 0.15, self.conf_dict[k], 0.0)

    def get_orig_results(self):
        return self.orig_results
