from sklearn.isotonic import IsotonicRegression
import bpr
from surprise import SVDpp
from pyfm import pylibfm


def build_ir():
    return IsotonicRegression()


def build_bpr():
    args = bpr.BPRArgs()
    args.learning_rate = 0.3
    num_factors = 10
    bpr_model = bpr.BPR(num_factors, args)
    sample_negative_items_empirically = True
    sampler = bpr.UniformPairWithoutReplacement(sample_negative_items_empirically)
    num_iters = 10
    return bpr_model, sampler, num_iters


def build_svdpp():
    return SVDpp()


def bulid_fm():
    return pylibfm.FM()
