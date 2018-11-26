from sklearn import linear_model
from sklearn import svm
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

classifiers = [
    ('SVM', svm.SVR()),
    ('SGD Regression', linear_model.SGDRegressor()),
    ('Bayesian Ridge Regression', linear_model.BayesianRidge()),
    ('Lasso Regression', linear_model.LassoLars()),
    ('Passive Aggressive Regression', linear_model.PassiveAggressiveRegressor()),
    ('Theil-Sen Regression', linear_model.TheilSenRegressor()),
    ('Linear Regression', linear_model.LinearRegression())]


def _avg(M):
    sumM = sum(sum(M))
    lenM = len(M) * len(M[0])
    avgM = sumM / lenM
    return avgM


def _max(M):
    sum_m = 0
    for i in M:
        sum_m += np.max(i)
    for j in M.T:
        sum_m += np.max(j)
    return sum_m / (len(M) + len(M.T))


def _std(M):
    return np.sqrt(np.var(M))


def _entropy(labels, base=None):
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent


def _mpe(M):
    sum_e = 0
    for i in M:
        sum_e += _entropy(i)
    for j in M.T:
        sum_e += _entropy(j)
    return sum_e / (len(M) + len(M[0]))


def _pca(M):
    pca = PCA()
    pca.fit(M)
    return [pca.singular_values_[0], pca.singular_values_[1], sum(pca.singular_values_), _entropy(pca.singular_values_)]


def _norms(M):
    return [np.linalg.norm(M, 1), np.linalg.norm(M, 2), np.linalg.norm(M, 'fro'), np.linalg.norm(M, np.inf)]


def _mcd(M):
    sum_ij = 0.0
    count_ij = 0.0
    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] > 0.0:
                mu_ij = (sum(M[i]) + sum(M.T[j]) - M[i][j]) / (len(M) + len(M[0]))
                sum_ij += math.pow((M[i][j] - mu_ij), 2)
            count_ij += 1
    if count_ij == 0.0:
        return 0
    return math.sqrt(sum_ij / count_ij)


def _dom(M):
    counter = 0.0
    for i in range(len(M)):
        max_r = np.max(M[i])
        for j in range(len(M[0])):
            max_c = np.max(M.T[j])
            if M[i][j] == max_r and M[i][j] == max_c:
                #                 print(M[i][j], max_r, max_c)
                counter += 1
    return counter / max(len(M), len(M[0]))


def _bpm(M):
    diff = 0.0
    for i in range(len(M)):
        for j in range(len(M[0])):
            if M[i][j] > 0.5:
                diff += (1.0 - M[i][j])
            else:
                diff += M[i][j]
    return diff


def _bbm(M):
    flat_M = M.reshape(len(M) * len(M[0])).reshape(1, -1)
    bin_M = np.zeros(flat_M.shape)
    for i in range(len(flat_M[0])):
        if flat_M[0][i] > 0.5:
            bin_M[0][i] = 1.0
    return cosine_similarity(flat_M, bin_M)[0][0]


def _lmm(M):
    counter = 0.0
    big_M = M
    if len(M) < len(M[0]):
        big_M = M.T
    for a in big_M:
        if 1.0 in list(a):
            counter += 1
    return counter / len(big_M)


def extractPreds(M):
    return np.array([_avg(M), _max(M), _std(M), _mpe(M), _mcd(M), _bbm(M), _bpm(M), _lmm(M), _dom(M)]
                    + _pca(M) + _norms(M)).reshape(1, -1)
