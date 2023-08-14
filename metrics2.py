import ipdb
import numpy as np
import tqdm
from scipy.stats import rankdata
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from scipy.stats import beta
from scipy.stats import percentileofscore
from scipy.optimize import fmin
from scipy.optimize import minimize_scalar
import scipy.integrate as integrate
# from betacal import BetaCalibration
from scipy.special import gamma as gamma_func
from scipy.stats import gamma

#from sklearn.metrics import brier_score_loss # Only for one-class
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

from sklearn.preprocessing import label_binarize


# To check of serializable objects
import json

import pickle
from utils import *


def guo_ECE(probs, y_true, bins=15):
    return ECE(probs, y_true, normalize=False, bins=bins, ece_full=False)

def ECE(probs, y_true, normalize=False, bins=15, ece_full=False):
    probs = np.array(probs)
    y_true = np.array(y_true)
    if len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1).reshape(-1, 1)

    # Prepare predictions, confidences and true labels for ECE calculation
    if ece_full:
        preds, confs, y_true = get_preds_all(probs, y_true, normalize=normalize, flatten=True)

    else:
        preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

        if normalize:
            confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(probs, axis=1)  # Take only maximum confidence


    # Calculate ECE and ECE2
    ece = ECE_helper(confs, preds, y_true, bin_size = 1/bins, ece_full = ece_full)

    return ece

def ECE_helper(conf, pred, true, bin_size = 0.1, ece_full = False):

    """
    Expected Calibration Error
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
    Returns:
        ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true, ece_full)
        # print(acc, avg_conf)
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE

    return ece

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true,
                    ece_full=True):
    """
    # Computes accuracy and average confidence for bin
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        pred_thresh (float) : float in range (0,1), indicating the prediction threshold
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if  (x[2] > conf_thresh_lower or conf_thresh_lower == 0)  and x[2] <= conf_thresh_upper]

    if len(filtered_tuples) < 1:
        return 0.0, 0.0, 0
    else:
        if ece_full:
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples])/len_bin  # Avg confidence of BIN
            accuracy = np.mean([x[1] for x in filtered_tuples])  # Mean difference from actual class
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
            accuracy = float(correct)/len_bin  # accuracy of BIN

    return accuracy, avg_conf, len_bin

def binary_ECE(probs, y_true, power = 1, bins = 15):

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1
    bin_func = lambda p, y, idx: (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power) * np.sum(idx) / len(probs)

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(probs, y_true, idx == i)
    return ece

def classwise_ECE(probs, y_true, power = 1, bins = 15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    n_classes = probs.shape[1]

    return np.mean(
        [
            binary_ECE(
                probs[:, c], y_true[:, c].astype(float), power = power, bins = bins
            ) for c in range(n_classes)
        ]
    )

def label_resampling(probs):
    c = probs.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    y = np.zeros_like(probs)
    y[range(len(probs)), choices] = 1
    return y


def score_sampling(probs, samples = 10000, ece_function = None):

    probs = np.array(probs)

    return np.array(
        [
            ece_function(probs, label_resampling(probs)) for sample in range(samples)
        ]
    )
    # lst = []
    # for sample in tqdm.tqdm(range(samples)):
    #     lst.append(ece_function(probs, label_resampling(probs)))
    #
    # return np.array(lst)

def pECE(probs, y_true, samples = 10000, ece_function = classwise_ECE):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    scores = score_sampling(probs,samples=samples,ece_function=ece_function)
    return 1 - (
        percentileofscore(
            scores,
            ece_function(probs, y_true)
        ) / 100
    )

if __name__ == '__main__':
    # FILE_PATH = '/home/ycliu/zouyl/uncertainty/logits/probs_resnet152_imgnet_logits.p'
    FILE_PATH = '/home/ycliu/zouyl/uncertainty/logits/probs_densenet40_c10_logits.p'
    (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(FILE_PATH, True)
    # softmax
    import torch
    y_probs_test = torch.nn.functional.softmax(torch.from_numpy(y_probs_test), dim=1).numpy()

    from sklearn.preprocessing import LabelBinarizer
    binarizer = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    y_val_bin = binarizer.fit_transform(y_val)
    y_test_bin = binarizer.transform(y_test)
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
    print(y_test_bin.shape)
    conf_ece = guo_ECE(y_probs_test, y_test)
    print('conf_ECE: %4f' % (conf_ece))
    cw_ece = classwise_ECE(y_probs_test, y_test)
    print('cw_ECE: %4f' % (cw_ece))
    # p_conf_ece = pECE(y_probs_test, y_test_bin, 10000, guo_ECE)
    # print('p_conf_ECE: %4f' % (p_conf_ece))
    # p_cw_ece = pECE(y_probs_test, y_test_bin, 10000, classwise_ECE)
    # print('p_cw_ECE: %4f' %(p_cw_ece))


    # acc, ece = ECE(y_probs_test, y_test)
    # acc, mce = MCE(y_probs_test, y_test)
    # nll = NLL(y_probs_test, y_test)
    # bs = BS(y_probs_test, y_test)
    # print(bs)
    # print('resnet110_SD_c10')
    # print('acc: %4f, ece: %4f, mce %4f, nll: %4f, bs: %4f' %(acc,ece,mce,nll,bs))