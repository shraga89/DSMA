import numpy as np
import pandas as pd
import py_entitymatching as em
import networkx as nx
# import matplotlib.pyplot as plt
# import os.path
import os
from shutil import copyfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
path = './ds_with_exact/Structured/dblp_scholar_exp_data/exp_data/'
import deepmatcher as dm


def get_features(c, t):
    feats = {}
    for index, row in features.iterrows():
        a_temp = A[A['id'] == c][row['left_attribute']].values[0]
        b_temp = B[B['id'] == t][row['left_attribute']].values[0]
        feat_name = row['feature_name']
        if a_temp == 'nan' or b_temp == 'nan':
            feats[feat_name] = 0
            continue
        if row['left_attr_tokenizer'] in ('qgm_3', 'dlm_dc0'):
            a_temp = em.feature.tokenizers.tok_qgram(input_string=a_temp, q=3)
            b_temp = em.feature.tokenizers.tok_qgram(input_string=b_temp, q=3)
        feats[feat_name] = sim[row['simfunction']](a_temp, b_temp)
    return feats


def get_predictions(c, t, predictions):
    preds = {}
    for p in predictions:
        temp = 0
        if len(predictions[p].loc[(predictions[p]['ltable_id'] == c) & (predictions[p]['rtable_id'] == t)][
                   'predicted'].index) > 0:
            temp = float(predictions[p].loc[(predictions[p]['ltable_id'] == c) & (predictions[p]['rtable_id'] == t)][
                             'predicted'])
        preds[p] = temp
    return preds


print(os.path.isfile(path + 'tableA.csv'))
A = em.read_csv_metadata(path + 'tableA.csv', key='id')
B = em.read_csv_metadata(path + 'tableB.csv', key='id')
train = pd.read_csv(path + 'train.csv', low_memory=False, encoding='ISO-8859-1')
test = pd.read_csv(path + 'test.csv', low_memory=False, encoding='ISO-8859-1')
valid = pd.read_csv(path + 'valid.csv', low_memory=False, encoding='ISO-8859-1')
frames = [train, test, valid]
exact = pd.concat(frames)
exact.columns = ['ltable.id', 'rtable.id', 'gold']
exact.to_csv(path + 'exact.csv', index_label='_id')

print('size of A: ', str(len(A)))
print('size of B: ', str(len(B)))
print('size of exact: ', str(len(exact)))

# trim by exact
# exact_l = list(exact['ltable.id'])
# exact_r = list(exact['rtable.id'])
# A = em.utils.pandas_helper.filter_rows(df = A, condn='id == {0}'.format(exact_l))
# B = em.utils.pandas_helper.filter_rows(df = B, condn='id == {0}'.format(exact_r))
# A = A[A['id'].isin(exact_l)]
# B = B[B['id'].isin(exact_r)]
# print('size of A: ', str(len(A)))
# print('size of B: ', str(len(B)))

ob = em.OverlapBlocker()

# K1 = ob.block_tables(A, B, 'name', 'name',
#                     l_output_attrs=list(A.columns), 
#                     r_output_attrs=list(B.columns),
#                     overlap_size=7)

# interest_cols = ['Song_Name', 'Artist_Name', 'Album_Name', 'Genre']
interest_cols = list(A.columns)

K1 = ob.block_tables(A, B, 'title', 'title',
                     l_output_attrs=interest_cols,
                     r_output_attrs=interest_cols,
                     overlap_size=5)
K1 = ob.block_candset(K1, 'authors', 'authors', overlap_size=3)

# K2 = ob.block_tables(A, B, 'Artist_Name', 'Artist_Name',
#                     l_output_attrs = interest_cols, 
#                     r_output_attrs = interest_cols,
#                     overlap_size=3)
# K2 = ob.block_candset(K2, 'Album_Name', 'Album_Name', overlap_size=3)

# K2 = ob.block_tables(A, B, 'description', 'description',
#                     l_output_attrs=list(A.columns),
#                     r_output_attrs=list(B.columns),
#                     overlap_size=8)

# K1 = ob.block_candset(K1, 'description', 'description', overlap_size=5)
# K3 = ob.block_candset(K1, 'price', 'price', overlap_size=1)
# K1 = em.combine_blocker_outputs_via_union([K1, K2])

G = nx.Graph()
G.add_nodes_from(K1['ltable_id'])
G.add_nodes_from(K1['rtable_id'])
G.add_edges_from(list(zip(K1['ltable_id'], K1['rtable_id'])))
blocks = list(nx.connected_components(G))

# G = nx.Graph()
# G.add_nodes_from(K2['ltable_id'])
# G.add_nodes_from(K2['rtable_id'])
# G.add_edges_from(list(zip(K2['ltable_id'], K2['rtable_id'])))
# blocks += list(nx.connected_components(G))

blocks = [b for b in blocks if 10 < len(b) < 100]
print("Num of blocks:" + str(len(blocks)))
print("Avg size of blocks:" + str(sum([len(b) for b in blocks]) / len(blocks)))
sim = em.get_sim_funs_for_matching()
features = em.get_features_for_matching(A.drop('id', axis=1), B.drop('id', axis=1))

dt = em.DTMatcher(name='DecisionTree')
svm = em.SVMMatcher(name='SVM')
rf = em.RFMatcher(name='RF')
nb = em.NBMatcher(name='NB')
lg = em.LogRegMatcher(name='LogReg')
ln = em.LinRegMatcher(name='LinReg')
matchers = [dt, svm, rf, nb, lg, ln]

# L = em.label_table(K1, 'gold', verbose=2)
# print(L.columns)
# print(type(L))
# print(K1.columns)
# print(type(K1))
# cols = list(K1.columns)
# cols[1:3] = ['ltable.id', 'rtable.id']
# K1.columns = cols
K1.to_csv(path + 'K1.csv')
L = em.read_csv_metadata(path + 'K1.csv',
                         key='_id',
                         ltable=A, rtable=B,
                         fk_ltable='ltable_id', fk_rtable='rtable_id')
# L = K1.copy()
# print(L.columns)
L['gold'] = 0
trues = exact[exact['gold'] == 1][['ltable.id', 'rtable.id']]
L['temp'] = L['ltable_id'].astype(str) + L['rtable_id'].astype(str)
trues['temp'] = trues['ltable.id'].astype(str) + trues['rtable.id'].astype(str)
L.loc[L['temp'].isin(trues['temp']), ['gold']] = 1

# true = list(exact[exact['gold'] == 1].index)
# L.loc[L['_id'].isin(true), ['gold']] = 1
# PERFORM JOIN!
# L = pd.merge(L.drop(columns='gold'), exact, how='inner', left_on=['ltable_id', 'rtable_id'],
# right_on=['ltable.id', 'rtable.id'])
# L = em.read_csv_metadata(path + 'exact.csv', 
#                          key='_id',
#                          ltable=A, rtable=B, 
#                          fk_ltable='ltable.id', fk_rtable='rtable.id')

development_evaluation = em.split_train_test(L, train_proportion=0.5)
development = development_evaluation['train']
evaluation = development_evaluation['test']

train_feature_vectors = em.extract_feature_vecs(development, attrs_after='gold',
                                                feature_table=features)
test_feature_vectors = em.extract_feature_vecs(evaluation, attrs_after='gold',
                                               feature_table=features)

train_feature_vectors = train_feature_vectors.fillna(0.0)
test_feature_vectors = test_feature_vectors.fillna(0.0)

print("tagged pairs:" + str(exact['gold'].value_counts()))

predictions = {}
for m in matchers:
    temp = test_feature_vectors.copy()
    m.fit(table=train_feature_vectors,
          exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'gold'],
          target_attr='gold')
    first = m.predict(table=temp,
                      exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'gold'],
                      append=True,
                      target_attr='predicted')
    m.__init__(name=m.get_name())
    temp = train_feature_vectors.copy()
    m.fit(table=test_feature_vectors,
          exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'gold'],
          target_attr='gold')
    second = m.predict(table=temp,
                       exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'gold'],
                       append=True,
                       target_attr='predicted')
    predictions[m.get_name()] = pd.concat([first, second])


if not os.path.exists(path + 'deep'):
    os.makedirs(path + 'deep')
L['label'] = L['gold']
L = L.drop(['Unnamed: 0', 'gold', 'temp'], axis=1)
print(L.columns)
dm.data.split(L, path + 'deep', 'train.csv', 'valid.csv', 'test.csv',
              [1, 1, 1])
if not os.path.exists(path + 'deep/unlabeled'):
    os.makedirs(path + 'deep/unlabeled')
temp = pd.read_csv(path + 'deep/train.csv')
temp = temp.drop(['label'], axis=1)
temp.to_csv(path + 'deep/unlabeled/train.csv')
temp = pd.read_csv(path + 'deep/test.csv')
temp = temp.drop(['label'], axis=1)
temp.to_csv(path + 'deep/unlabeled/test.csv')
temp = pd.read_csv(path + 'deep/valid.csv')
temp = temp.drop(['label'], axis=1)
temp.to_csv(path + 'deep/unlabeled/valid.csv')
train, validation, test = dm.data.process(
    path=path + 'deep',
    cache='train_cache.pth',
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    use_magellan_convention=True,
    ignore_columns=('ltable_id', 'rtable_id'))
deepModel = dm.MatchingModel()
deepModel.run_train(
    train,
    validation,
    best_save_path='best_model.pth')
unlabeled = dm.data.process_unlabeled(path=path + 'deep/unlabeled' + '/test.csv', trained_model=deepModel)
first = deepModel.run_prediction(unlabeled)
# print(deepModel)
print(first)
del deepModel
train, validation, test = dm.data.process(
    path=path + 'deep',
    cache='train_cache.pth',
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    use_magellan_convention=True,
    ignore_columns=('ltable_id', 'rtable_id'))
deepModel = dm.MatchingModel()
deepModel.run_train(
    validation,
    test,
    best_save_path='best_model1.pth')
unlabeled = dm.data.process_unlabeled(path=path + 'deep/unlabeled' + '/train.csv', trained_model=deepModel)
second = deepModel.run_prediction(unlabeled)
print(second)
del deepModel
train, validation, test = dm.data.process(
    path=path + 'deep',
    cache='train_cache.pth',
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    use_magellan_convention=True,
    ignore_columns=('ltable_id', 'rtable_id'))
deepModel = dm.MatchingModel()
deepModel.run_train(
    test,
    train,
    best_save_path='best_model2.pth')
unlabeled = dm.data.process_unlabeled(path=path + 'deep/unlabeled' + '/valid.csv', trained_model=deepModel)
third = deepModel.run_prediction(unlabeled)
predictions["deepMatcher"] = pd.concat([first, second, third])
print(predictions["deepMatcher"])

df = pd.DataFrame(columns=['instance', 'candName', 'targName', 'conf', 'realConf'])
epoch = 1
for block in range(len(blocks)):
    print('Creating Block Number: ' + str(block))
    cands = []
    targs = []
    for i in list(blocks[block]):
        if i in list(A['id']):
            cands.append(i)
        if i in list(B['id']):
            targs.append(i)
    for c in cands:
        for t in targs:
            e = 0
            if len(exact[(exact['ltable.id'] == c) & (exact['rtable.id'] == t)]['gold'].index) > 0:
                e = float(exact[(exact['ltable.id'] == c) & (exact['rtable.id'] == t)]['gold'])
            feat = get_features(c, t)
            for f in feat:
                res_row = np.concatenate((np.array(str(block) + ' ' + str(f)), np.array(str(c)),
                                          np.array(str(t)), np.array(feat[f]), np.array(e)), axis=None)
                df.loc[epoch] = res_row
                epoch += 1
            pred = get_predictions(c, t, predictions)
            for p in pred:
                res_row = np.concatenate((np.array(str(block) + ' ' + str(p)), np.array(str(c)),
                                          np.array(str(t)), np.array(pred[p]), np.array(e)), axis=None)
                df.loc[epoch] = res_row
                epoch += 1
df.to_csv(path + 'em_dataset.csv', index=False)
