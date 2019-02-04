import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support
from surprise import Dataset
from surprise import Reader
import pandas as pd
from scipy import sparse
import FeatureBasedEvaluator as FBE
from sklearn.feature_extraction import DictVectorizer


def deep_adapt_and_evaluate(instance, _type, adaptor, is_adaptor_mat, evaluator, is_evaluator_mat, X_seq, X_mat, y,
                            y_single,
                            res_adapt_eval, count_adapt_eval):
    if is_adaptor_mat:
        yhat_full = adaptor.predict_classes(X_mat, verbose=2)
    else:
        yhat_full = adaptor.predict_classes(X_seq, verbose=2)
    yhat_full = np.array(yhat_full.reshape(yhat_full.shape[1:-1] + (1,)))
    if is_evaluator_mat:
        yhat_single = evaluator.predict(X_mat, verbose=2)
    else:
        yhat_single = evaluator.predict(X_seq, verbose=2)
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y,
                                                                    np.ceil(np.array(X_seq.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                    np.array(y_single), np.array(yhat_single)), axis=None)
    res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
    count_adapt_eval += 1
    k_adapt += 1
    yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
    yhat_new = evaluator.predict(yhat_full, verbose=2)
    while yhat_new > yhat_single:
        if is_adaptor_mat:
            X_mat = yhat_full
            yhat_full = adaptor.predict_classes(X_mat, verbose=2)
        else:
            X_seq = yhat_full
            yhat_full = adaptor.predict_classes(X_seq, verbose=2)
        yhat_full = np.array(yhat_full.reshape(yhat_full.shape[1:-1] + (1,)))
        yhat_single = yhat_new
        res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X_seq.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        if is_evaluator_mat:
            yhat_new = evaluator.predict(X_mat, verbose=2)
        else:
            yhat_new = evaluator.predict(X_seq, verbose=2)
    return res_adapt_eval, count_adapt_eval


def deep_adapt_and_evaluate_multi(instance, _type, multi, X_seq, y, y_single, res_adapt_eval, count_adapt_eval):
    predicted = multi.predict(X_seq, verbose=2)
    yhat_full = predicted[0]
    yhat_full = np.where(yhat_full < 0.5, 0.0, 1.0)
    yhat_full = np.array(yhat_full.reshape(yhat_full.shape[1:-1] + (1,)))
    yhat_single = predicted[1]
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y,
                                                                    np.ceil(np.array(X_seq.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                    np.array(y_single), np.array(yhat_single)), axis=None)
    res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
    count_adapt_eval += 1
    k_adapt += 1
    yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
    predicted = multi.predict(yhat_full, verbose=2)
    yhat_new = predicted[1]
    while yhat_new > yhat_single:
        X_seq = yhat_full
        predicted = multi.predict(X_seq, verbose=2)
        yhat_full = predicted[0]
        yhat_full = np.where(yhat_full < 0.5, 0.0, 1.0)
        yhat_full = np.array(yhat_full.reshape(yhat_full.shape[1:-1] + (1,)))
        yhat_single = yhat_new
        res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X_seq.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        yhat_new = predicted[1]
    return res_adapt_eval, count_adapt_eval


def deep_adapt_and_reg_evaluate(instance, _type, adaptor, evaluators, X, matN, matM, X_f, y, y_single, res_adapt_eval,
                                count_adapt_eval):
    X_orig = X
    for clf in evaluators:
        X = X_orig
        yhat_full = None
        yhat_full = adaptor.predict_classes(X, verbose=2)
        yhat_full = np.array(yhat_full.reshape(yhat_full.shape[1:-1] + (1,)))
        yhat_single = clf[1].predict(X=np.nan_to_num(X_f))
        k_adapt = 0
        res_row_adapt = np.concatenate((instance, _type + clf[0], str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                    np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
        yhat_new = clf[1].predict(X=np.nan_to_num(FBE.extractPreds(yhat_full.reshape(matN, matM))))
        while yhat_new > yhat_single:
            X = yhat_full
            yhat_full = adaptor.predict_classes(X, verbose=2)
            yhat_full = np.array(yhat_full.reshape(yhat_full.shape[1:-1] + (1,)))
            yhat_single = yhat_new
            res_row_adapt = np.concatenate((instance, _type + clf[0], str(k_adapt),
                                            precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                            precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                            np.array(y_single), np.array(yhat_single)), axis=None)
            res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
            count_adapt_eval += 1
            k_adapt += 1
            yhat_new = clf[1].predict(X=np.nan_to_num(FBE.extractPreds(X.reshape(matN, matM))))
    return res_adapt_eval, count_adapt_eval


def ir_adapt_and_deep_evaluate(instance, _type, adaptor, evaluator, X, y, y_single, res_adapt_eval, count_adapt_eval):
    try:
        yhat_full = adaptor.predict(X.reshape(X.shape[1]))
    except:
        yhat_full = np.ceil(np.array(X.reshape(y.shape)))
    yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
    yhat_full[np.isnan(yhat_full)] = 0
    yhat_single = evaluator.predict(X, verbose=2)
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y,
                                                                    np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                    np.array(y_single), np.array(yhat_single)), axis=None)
    res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
    count_adapt_eval += 1
    k_adapt += 1
    yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
    yhat_new = evaluator.predict(yhat_full, verbose=2)
    while yhat_new > yhat_single:
        X = yhat_full
        try:
            yhat_full = adaptor.predict(X.reshape(X.shape[1]))
        except:
            yhat_full = np.ceil(np.array(X.reshape(y.shape)))
        yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
        yhat_full[np.isnan(yhat_full)] = 0
        yhat_single = evaluator.predict(X, verbose=2)
        yhat_single = yhat_new
        res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        yhat_new = evaluator.predict(X, verbose=2)
    return res_adapt_eval, count_adapt_eval


def svd_adapt_and_deep_evaluate(instance, _type, adaptor, evaluator, X, size_m, size_n,
                                y, y_single, res_adapt_eval, count_adapt_eval):
    items = list()
    for j in range(size_m):
        items += [j] * size_n
    users = list(range(size_n)) * size_m
    ratings_dict = {'itemID': items,
                    'userID': users,
                    'rating': X.reshape(X.shape[1])}
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0.0, 1.0))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    x = data.build_full_trainset()
    adaptor.fit(x)
    yhat_full = list()
    test = adaptor.test(x.build_testset())
    for t in test:
        yhat_full += [t[2]]
    yhat_full = np.array(yhat_full)
    yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
    yhat_single = evaluator.predict(X, verbose=2)
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y,
                                                                    np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                    np.array(y_single), np.array(yhat_single)), axis=None)
    res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
    count_adapt_eval += 1
    k_adapt += 1
    yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
    yhat_new = evaluator.predict(yhat_full, verbose=2)
    while yhat_new > yhat_single:
        X = yhat_full
        items = list()
        for j in range(size_m):
            items += [j] * size_n
        users = list(range(size_n)) * size_m
        ratings_dict = {'itemID': items,
                        'userID': users,
                        'rating': X.reshape(X.shape[1])}
        df = pd.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(0.0, 1.0))
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        x = data.build_full_trainset()
        adaptor.fit(x)
        yhat_full = list()
        test = adaptor.test(x.build_testset())
        for t in test:
            yhat_full += [t[2]]
        yhat_full = np.array(yhat_full)
        yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
        yhat_single = evaluator.predict(X, verbose=2)
        yhat_single = yhat_new
        res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        yhat_new = evaluator.predict(X, verbose=2)
    return res_adapt_eval, count_adapt_eval


def bpr_adapt_and_deep_evaluate(instance, _type, adaptor, evaluator, X, matN, matM,
                                y, y_single, res_adapt_eval, count_adapt_eval):
    try:
        X_mat = X[1].reshape(matN, matM)
        yhat_full = list()
        for u in range(X_mat.shape[0]):
            for i in range(X_mat.shape[1]):
                try:
                    yhat_full += [adaptor[0].predict(i=i, u=u)]
                except:
                    yhat_full += [0.0]
        yhat_full = np.array(yhat_full)
        yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
    except:
        yhat_full = np.ceil(np.array(X.reshape(y.shape)))
    yhat_single = evaluator.predict(X, verbose=2)
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y,
                                                                    np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                    np.array(y_single), np.array(yhat_single)), axis=None)
    res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
    count_adapt_eval += 1
    k_adapt += 1
    yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
    yhat_new = evaluator.predict(yhat_full, verbose=2)
    while yhat_new > yhat_single:
        X = yhat_full
        try:
            X_mat = X[1].reshape(matN, matM)
            yhat_full = list()
            for u in range(X_mat.shape[0]):
                for i in range(X_mat.shape[1]):
                    try:
                        yhat_full += [adaptor[0].predict(i=i, u=u)]
                    except:
                        yhat_full += [0.0]
            yhat_full = np.array(yhat_full)
            yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
        except:
            yhat_full = np.ceil(np.array(X.reshape(y.shape)))
        yhat_single = evaluator.predict(X, verbose=2)
        yhat_single = yhat_new
        res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        yhat_new = evaluator.predict(X, verbose=2)
    return res_adapt_eval, count_adapt_eval


def ir_adapt_and_reg_evaluate(instance, _type, adaptor, evaluators, X, matN, matM, X_f, y, y_single,
                                    res_adapt_eval, count_adapt_eval):
    X_orig = X
    for clf in evaluators:
        X = X_orig
        try:
            yhat_full = adaptor.predict(X.reshape(X.shape[1]))
        except:
            yhat_full = np.ceil(np.array(X.reshape(y.shape)))
        yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
        yhat_full[np.isnan(yhat_full)] = 0
        yhat_single = clf[1].predict(X=np.nan_to_num(X_f))
        k_adapt = 0
        res_row_adapt = np.concatenate((instance, _type + clf[0], str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
        yhat_new = clf[1].predict(X=np.nan_to_num(FBE.extractPreds(yhat_full.reshape(matN, matM))))
        while yhat_new > yhat_single:
            X = yhat_full
            try:
                yhat_full = adaptor.predict(X.reshape(X.shape[1]))
            except:
                yhat_full = np.ceil(np.array(X.reshape(y.shape)))
            yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
            yhat_full[np.isnan(yhat_full)] = 0
            yhat_single = yhat_new
            res_row_adapt = np.concatenate((instance, _type + clf[0], str(k_adapt),
                                            precision_recall_fscore_support(y,
                                                                            np.ceil(
                                                                                np.array(
                                                                                    X.reshape(yhat_full.shape))),
                                                                            average='binary')[:3],
                                            precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                            np.array(y_single), np.array(yhat_single)), axis=None)
            res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
            count_adapt_eval += 1
            k_adapt += 1
            yhat_new = clf[1].predict(X=np.nan_to_num(FBE.extractPreds(X.reshape(matN, matM))))
    return res_adapt_eval, count_adapt_eval


def svd_adapt_and_reg_evaluate(instance, _type, adaptor, evaluators, X, matN, matM, X_f, y, y_single,
                                    res_adapt_eval, count_adapt_eval):
    X_orig = X
    for clf in evaluators:
        X = X_orig
        items = list()
        for j in range(matM):
            items += [j] * matN
        users = list(range(matN)) * matM
        ratings_dict = {'itemID': items,
                        'userID': users,
                        'rating': X.reshape(X.shape[1])}
        df = pd.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(0.0, 1.0))
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        x = data.build_full_trainset()
        adaptor.fit(x)
        yhat_full = list()
        test = adaptor.test(x.build_testset())
        for t in test:
            yhat_full += [t[2]]
        yhat_full = np.array(yhat_full)
        yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
        yhat_single = clf[1].predict(X=np.nan_to_num(X_f))
        k_adapt = 0
        res_row_adapt = np.concatenate((instance, _type + clf[0], str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
        yhat_new = clf[1].predict(X=np.nan_to_num(FBE.extractPreds(yhat_full.reshape(matN, matM))))
        while yhat_new > yhat_single:
            X = yhat_full
            items = list()
            for j in range(matM):
                items += [j] * matN
            users = list(range(matN)) * matM
            ratings_dict = {'itemID': items,
                            'userID': users,
                            'rating': X.reshape(X.shape[1])}
            df = pd.DataFrame(ratings_dict)
            reader = Reader(rating_scale=(0.0, 1.0))
            data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
            x = data.build_full_trainset()
            adaptor.fit(x)
            yhat_full = list()
            test = adaptor.test(x.build_testset())
            for t in test:
                yhat_full += [t[2]]
            yhat_full = np.array(yhat_full)
            yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
            yhat_single = yhat_new
            res_row_adapt = np.concatenate((instance, _type + clf[0], str(k_adapt),
                                            precision_recall_fscore_support(y,
                                                                            np.ceil(
                                                                                np.array(
                                                                                    X.reshape(yhat_full.shape))),
                                                                            average='binary')[:3],
                                            precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                            np.array(y_single), np.array(yhat_single)), axis=None)
            res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
            count_adapt_eval += 1
            k_adapt += 1
            yhat_new = clf[1].predict(X=np.nan_to_num(FBE.extractPreds(X.reshape(matN, matM))))
    return res_adapt_eval, count_adapt_eval


def bpr_adapt_and_reg_evaluate(instance, _type, adaptor, evaluators, X, matN, matM, X_f, y, y_single,
                                    res_adapt_eval, count_adapt_eval):
    X_orig = X
    for clf in evaluators:
        X = X_orig
        try:
            X_mat = X[1].reshape(matN, matM)
            yhat_full = list()
            for u in range(X_mat.shape[0]):
                for i in range(X_mat.shape[1]):
                    try:
                        yhat_full += [adaptor[0].predict(i=i, u=u)]
                    except:
                        yhat_full += [0.0]
            yhat_full = np.array(yhat_full)
            yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
        except:
            yhat_full = np.ceil(np.array(X.reshape(y.shape)))
        yhat_single = clf[1].predict(X=np.nan_to_num(X_f))
        k_adapt = 0
        res_row_adapt = np.concatenate((instance, _type + clf[0], str(k_adapt),
                                        precision_recall_fscore_support(y,
                                                                        np.ceil(
                                                                            np.array(X.reshape(yhat_full.shape))),
                                                                        average='binary')[:3],
                                        precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                        np.array(y_single), np.array(yhat_single)), axis=None)
        res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
        count_adapt_eval += 1
        k_adapt += 1
        yhat_full = np.array(yhat_full.reshape((1,) + yhat_full.shape))
        yhat_new = clf[1].predict(X=np.nan_to_num(FBE.extractPreds(yhat_full.reshape(matN, matM))))
        while yhat_new > yhat_single:
            X = yhat_full
            try:
                X_mat = X[1].reshape(matN, matM)
                yhat_full = list()
                for u in range(X_mat.shape[0]):
                    for i in range(X_mat.shape[1]):
                        try:
                            yhat_full += [adaptor[0].predict(i=i, u=u)]
                        except:
                            yhat_full += [0.0]
                yhat_full = np.array(yhat_full)
                yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
            except:
                yhat_full = np.ceil(np.array(X.reshape(y.shape)))
            yhat_single = yhat_new
            res_row_adapt = np.concatenate((instance, _type + clf[0], str(k_adapt),
                                            precision_recall_fscore_support(y,
                                                                            np.ceil(
                                                                                np.array(
                                                                                    X.reshape(yhat_full.shape))),
                                                                            average='binary')[:3],
                                            precision_recall_fscore_support(y, yhat_full, average='binary')[:3],
                                            np.array(y_single), np.array(yhat_single)), axis=None)
            res_adapt_eval.loc[count_adapt_eval] = res_row_adapt
            count_adapt_eval += 1
            k_adapt += 1
            yhat_new = clf[1].predict(X=np.nan_to_num(FBE.extractPreds(X.reshape(matN, matM))))
    return res_adapt_eval, count_adapt_eval


def only_deep_evaluate(instance, _type, X, y, evaluator, res_eval, count_eval, cos):
    yhat_single = evaluator.predict(X, verbose=2)
    res_row_eval = np.concatenate(
        (instance, _type, '-', np.array(y), np.array(yhat_single), np.array(np.sum(X)), np.array(cos)), axis=None)
    res_eval.loc[count_eval] = res_row_eval
    count_eval += 1
    return res_eval, count_eval


def only_deep_evaluate_multi(instance, _type, X, y, evaluator, res_eval, count_eval, cos):
    # print(evaluator.predict(X, verbose=2))
    yhat_single = evaluator.predict(X, verbose=2)[1]
    res_row_eval = np.concatenate(
        (instance, _type, '-', np.array(y), np.array(yhat_single), np.array(np.sum(X)), np.array(cos)), axis=None)
    res_eval.loc[count_eval] = res_row_eval
    count_eval += 1
    return res_eval, count_eval


def only_deep_adapt(instance, _type, X, y, adaptor, res_adapt, count_adapt):
    yhat_full = adaptor.predict_classes(X, verbose=2)
    yhat_full = np.array(yhat_full.reshape(yhat_full.shape[1:-1] + (1,)))
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y, np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3]),
                                   axis=None)
    res_adapt.loc[count_adapt] = res_row_adapt
    count_adapt += 1
    return res_adapt, count_adapt


def only_deep_adapt_multi(instance, _type, X, y, adaptor, res_adapt, count_adapt):
    # print(adaptor.predict(X, verbose=2))
    yhat_full = adaptor.predict(X, verbose=2)[0]
    yhat_full = np.where(yhat_full < 0.5, 0.0, 1.0)
    yhat_full = np.array(yhat_full.reshape(yhat_full.shape[1:-1] + (1,)))
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y, np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3]),
                                   axis=None)
    res_adapt.loc[count_adapt] = res_row_adapt
    count_adapt += 1
    return res_adapt, count_adapt


def only_reg_evaluate(instance, _type, X, y, evaluators, res_eval, count_eval):
    for clf in evaluators:
        yhat_single = clf[1].predict(X=np.nan_to_num(X))
        res_row_eval = np.concatenate(
            (instance, _type + clf[0], '-', np.array(y), np.array(yhat_single), np.array([0.0]), np.array([0.0])), axis=None)
        res_eval.loc[count_eval] = res_row_eval
        count_eval += 1
    return res_eval, count_eval


def reg_adapt_ir(instance, _type, X, y, adaptor, res_adapt, count_adapt):
    X = X.reshape(X.shape[1])
    try:
        yhat_full = adaptor.predict(X)
    except:
        yhat_full = np.ceil(np.array(X.reshape(y.shape)))
    yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
    k_adapt = 0
    yhat_full[np.isnan(yhat_full)] = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y, np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3]),
                                   axis=None)
    res_adapt.loc[count_adapt] = res_row_adapt
    count_adapt += 1
    return res_adapt, count_adapt


def reg_adapt_svd(instance, _type, X, y, adaptor, size_m, size_n, res_adapt, count_adapt):
    X = X.reshape(X.shape[1])
    items = list()
    for j in range(size_m):
        items += [j] * size_n
    users = list(range(size_n)) * size_m
    ratings_dict = {'itemID': items,
                    'userID': users,
                    'rating': X}
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0.0, 1.0))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    x = data.build_full_trainset()
    adaptor.fit(x)
    yhat_full = list()
    test = adaptor.test(x.build_testset())
    for t in test:
        yhat_full += [t[2]]
    yhat_full = np.array(yhat_full)
    yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))

    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y, np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3]),
                                   axis=None)
    res_adapt.loc[count_adapt] = res_row_adapt
    count_adapt += 1
    return res_adapt, count_adapt


def loadData(x_mat):
    data = []
    y = []
    users = set()
    items = set()
    for ui, u in enumerate(x_mat):
        for ii, i in enumerate(u):
            data.append({"user_id": str(ui), "movie_id": str(ii)})
            y.append(float(x_mat[ui, ii]))
            users.add(ui)
            items.add(ii)
    return (data, np.array(y), users, items)


def reg_adapt_fm(instance, _type, X, y, adaptor, size_m, size_n, res_adapt, count_adapt):

    (test_data, y_test, test_users, test_items) = loadData(X[0].reshape(X.shape[1:3]))
    v = DictVectorizer()
    X_test = v.fit_transform(test_data)
    yhat_full = adaptor.predict(X_test)
    yhat_full[np.isnan(yhat_full)] = np.ceil(np.array(X.reshape(yhat_full.shape)))[np.isnan(yhat_full)]
    yhat_full = np.array(yhat_full)
    yhat_full = np.array(yhat_full.reshape(len(yhat_full), 1)).round()
    # print(yhat_full.dtype)
    # if ((yhat_full != 0.) & (yhat_full != 1.)).any():
    #     print(yhat_full[(yhat_full != 1) & (yhat_full != 0)])
    #     print(yhat_full[(yhat_full != 1) & (yhat_full != 0)].index)
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y, np.ceil(np.array(X.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3]),
                                   axis=None)
    res_adapt.loc[count_adapt] = res_row_adapt
    count_adapt += 1
    return res_adapt, count_adapt


def reg_adapt_bpr(instance, _type, X, X_seq, y, adaptor, res_adapt, count_adapt):
    try:
        adaptor[0].train(sparse.csr_matrix(X.reshape(X.shape[1:3])), adaptor[1], adaptor[2])
        yhat_full = list()
        for u in range(X.shape[1]):
            for i in range(X.shape[2]):
                try:
                    yhat_full += [adaptor[0].predict(i=i, u=u)]
                except:
                    yhat_full += [0.0]

        yhat_full = np.array(yhat_full)
        yhat_full = np.round(np.array(yhat_full.reshape(len(yhat_full), 1)))
    except:
        yhat_full = np.ceil(np.array(X.reshape(y.shape)))
    k_adapt = 0
    res_row_adapt = np.concatenate((instance, _type, str(k_adapt),
                                    precision_recall_fscore_support(y,
                                                                    np.ceil(np.array(X_seq.reshape(yhat_full.shape))),
                                                                    average='binary')[:3],
                                    precision_recall_fscore_support(y, yhat_full, average='binary')[:3]),
                                   axis=None)
    res_adapt.loc[count_adapt] = res_row_adapt
    count_adapt += 1
    return res_adapt, count_adapt


def summerize(data, file):
    if 'pred_e' in data:
        idMax = data.groupby(['type'])['pred_e'].transform(max) == data['pred_e']
        pd.DataFrame(data[idMax].groupby(by='type').mean()).to_csv(file)
    else:
        pd.DataFrame(data.groupby(by='type').mean()).to_csv(file)