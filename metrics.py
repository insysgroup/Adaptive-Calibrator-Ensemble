import fire
import numpy as np

import torch
import torchvision as tv
import torchvision
import os
import ipdb
import tqdm
import time
import logging

from utils import *

class _ECELoss():
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def eval(self, confidences, accuracies):
        ece = np.zeros(1)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.__gt__(bin_lower) * confidences.__le__(bin_upper)
            prop_in_bin = in_bin.astype(float).mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

def ECE(logits, labels, n_bins=25, isLogits=0):
    # logits : logits of model's prediction (n_sample * n_class)
    # labels : ground truth (n_samples, )
    ece_criterion = _ECELoss(n_bins)
    if isLogits == 0:
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
    elif isLogits == 1:
        # input is softmax(logits)
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
    elif isLogits == 2:
        # input is confidences-->argmax(softmax(logits))
        # ipdb.set_trace()
        # confidences = logits
        # accuracies = labels
        acc = labels.sum()/len(labels)
        ece = ece_criterion.eval(logits.cpu().numpy(), labels.cpu().numpy())
        return acc,ece

    acc = accuracies.float().sum()/len(accuracies)
    ece = ece_criterion.eval(confidences.cpu().numpy(), accuracies.cpu().numpy())
    return acc, ece

class _MCELoss():
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_MCELoss, self).__init__()
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def eval(self, confidences, accuracies):
        mce = np.zeros(1)
        ce = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.__gt__(bin_lower) * confidences.__le__(bin_upper)
            prop_in_bin = in_bin.astype(float).mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ce.append(np.abs(avg_confidence_in_bin - accuracy_in_bin))
        mce[0] = max(ce)
        return mce

def MCE(logits, labels, n_bins=25, isLogits=0):
    # logits : logits of model's prediction (n_sample * n_class)
    # labels : ground truth (n_samples, )
    # all the inputs should be cuda tensor
    ece_criterion = _MCELoss(n_bins)
    if isLogits==0:
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
    elif isLogits==1:
        # input is softmax(logits)
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
    elif isLogits==2:
        # input is confidences-->argmax(softmax(logits))
        confidences = logits
        accuracies = labels

    acc = accuracies.float().sum()/len(accuracies)
    ece = ece_criterion.eval(confidences.cpu().numpy(), accuracies.cpu().numpy())
    return acc, ece

def AvgConf(logits, labels, isLogits=0):
    if isLogits==0:
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
    elif isLogits==1:
        # input is softmax(logits)
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
    elif isLogits==2:
        # input is confidences-->argmax(softmax(logits))
        confidences = logits
        accuracies = labels
    acc = accuracies.float().sum()/len(accuracies)
    avg_conf = confidences.cpu().numpy().mean()
    return avg_conf, acc


def NLL(logits, labels, isLogits=0):
    # logits : logits of model's prediction (n_sample * n_class)
    # labels : ground truth (n_samples, )
    # --------->only confidence can not calculate NLL
    if isLogits == 0:
        nll_criterion = torch.nn.CrossEntropyLoss().cuda()
        nll = nll_criterion(logits, labels).item()
    elif isLogits == 1:
        # input is softmax(logits)
        nll_criterion = torch.nn.NLLLoss().cuda()
        log = -torch.log(logits)
        nll = nll_criterion(logits, labels).item()
    return nll

def BS(logits, labels, isLogits=0):
    # logits : logits of model's prediction (n_sample * n_class)
    # labels : ground truth (n_samples, )
    # --------->only confidence can not calculate brier score
    labels_one_hot = torch.zeros(len(labels), logits.shape[-1]).cuda().scatter_(1, labels.reshape(-1,1), 1)
    if isLogits==0:
        softmaxs = torch.nn.functional.softmax(logits, dim=1)
    elif isLogits==1:
        # input is softmax(logits)
        softmaxs = logits
    bs = np.power(softmaxs.cpu().numpy()-labels_one_hot.cpu().numpy(), 2).sum() / (softmaxs.shape[0]*softmaxs.shape[1])
    return bs
    # from sklearn.metrics import mean_squared_error
    # labels_one_hot = torch.zeros(len(labels), logits.shape[-1]).cuda().scatter_(1, labels.reshape(-1, 1), 1)
    # softmaxs = torch.nn.functional.softmax(logits, dim=1)
    # return mean_squared_error(labels_one_hot.cpu().numpy(), softmaxs.cpu().numpy())

if __name__ == '__main__':
    # FILE_PATH = '/home/ycliu/zouyl/uncertainty/logits/probs_resnet110_SD_c10_logits.p'
    # (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(FILE_PATH, True)
    # y_probs_test = torch.from_numpy(y_probs_test).cuda()
    # y_test = torch.from_numpy(np.squeeze(y_test)).cuda()
    # acc, ece = ECE(y_probs_test, y_test)
    # acc, mce = MCE(y_probs_test, y_test)
    # nll = NLL(y_probs_test, y_test)
    # bs = BS(y_probs_test, y_test)
    # # print(bs)
    # print('resnet110_SD_c10')
    # print('acc: %4f, ece: %4f, mce %4f, nll: %4f, bs: %4f' %(acc,ece,mce,nll,bs))
    #
    # FILE_PATH = '/home/ycliu/zouyl/uncertainty/logits/probs_resnet110_SD_c100_logits.p'
    # (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(FILE_PATH, True)
    # y_probs_test = torch.from_numpy(y_probs_test).cuda()
    # y_test = torch.from_numpy(np.squeeze(y_test)).cuda()
    # acc, ece = ECE(y_probs_test, y_test)
    # acc, mce = MCE(y_probs_test, y_test)
    # nll = NLL(y_probs_test, y_test)
    # bs = BS(y_probs_test, y_test)
    # print('resnet110_SD_c100')
    # print('acc: %4f, ece: %4f, mce %4f, nll: %4f, bs: %4f' % (acc, ece, mce, nll, bs))

    files_10 = ('probs_resnet_wide32_c10_logits.p', 'probs_densenet40_c10_logits.p',
                'probs_lenet5_c10_logits.p', 'probs_resnet110_SD_c10_logits.p',
                'probs_resnet110_c10_logits.p')
    files_100 = ('probs_resnet_wide32_c100_logits.p', 'probs_densenet40_c100_logits.p',
                 'probs_lenet5_c100_logits.p', 'probs_resnet110_SD_c100_logits.p',
                 'probs_resnet110_c100_logits.p')
    for f in files_10:
        print(f)
        FILE_PATH = os.path.join('/home/ycliu/zouyl/uncertainty/logits', f)
        (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(FILE_PATH, True)
        y_probs_test = torch.from_numpy(y_probs_test).cuda()
        y_test = torch.from_numpy(np.squeeze(y_test)).cuda()
        acc, ece = ECE(y_probs_test, y_test)
        acc, mce = MCE(y_probs_test, y_test)
        nll = NLL(y_probs_test, y_test)
        bs = BS(y_probs_test, y_test)

        print('acc: %4f, ece: %4f, mce %4f, nll: %4f, bs: %4f' %(acc,ece,mce,nll,bs))
    for f in files_100:
        print(f)
        FILE_PATH = os.path.join('/home/ycliu/zouyl/uncertainty/logits', f)
        (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(FILE_PATH, True)
        y_probs_test = torch.from_numpy(y_probs_test).cuda()
        y_test = torch.from_numpy(np.squeeze(y_test)).cuda()
        acc, ece = ECE(y_probs_test, y_test)
        acc, mce = MCE(y_probs_test, y_test)
        nll = NLL(y_probs_test, y_test)
        bs = BS(y_probs_test, y_test)

        print('acc: %4f, ece: %4f, mce %4f, nll: %4f, bs: %4f' %(acc,ece,mce,nll,bs))