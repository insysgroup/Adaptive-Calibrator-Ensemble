from __future__ import absolute_import, division, print_function, unicode_literals

import functools

import ipdb
import numpy as np
import sys
import os

import torch
import yaml

# import matplotlib.pyplot as plt
import pickle
# from cal_metrics.ECE import _ECELoss
# from cal_metrics.MCE import _MCELoss
# from cal_metrics.KS import plot_KS_graphs
# from log_utils import ResultsLog
from scipy.special import softmax

from . import splines_utils as utils
from .splines_utils.KS import *

# np.random.seed(333)

def ensure_numpy(a):
  if not isinstance(a, np.ndarray): a = a.numpy()
  return a

def unpickle_probs(fname):
  # Read and open the file
  with open(fname, 'rb') as f:
    (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)

  y_probs_val, y_probs_test = softmax(y_probs_val, 1), softmax(y_probs_test, 1)
  return ((y_probs_val, y_val), (y_probs_test, y_test))

class interpolated_function :

  def __init__ (self, x, y) :
    self.x = x
    self.y = y
    self.lastindex = utils.len0(self.x)-1
    self.low = self.x[0]
    self.high = self.x[-1]


  def __call__ (self, x) :
    # Finds the interpolated value of the function at x

    # Easiest thing if value is out of range is to give maximum value
    if x >= self.x[-1] : return self.y[-1]
    if x <= self.x[0]  : return self.y[0]

    # Find the first x above.  ind cannot be 0, because of previous test
    # ind cannot be > lastindex, because of last test
    ind = first_above (self.x, x)

    alpha = x - self.x[ind-1]
    beta  = self.x[ind] - x

    # Special case.  This occurs when two values of x are equal
    if alpha + beta == 0 :
      return y[ind]

    return float((beta * self.y[ind] + alpha * self.y[ind-1]) / (alpha + beta))

#------------------------------------------------------------------------------


def get_recalibration_function(scores_in, labels_in, spline_method, splines, title=None) :
  # Find a function for recalibration

  # Change to numpy
  scores = ensure_numpy (scores_in)
  labels = ensure_numpy (labels_in)

  # Sort the data according to score
  order = scores.argsort()
  scores = scores[order]
  labels = labels[order]

  #Accumulate and normalize by dividing by num samples
  nsamples = utils.len0(scores)
  integrated_accuracy = np.cumsum(labels) / nsamples
  integrated_scores = np.cumsum(scores) / nsamples
  percentile = np.linspace (0.0, 1.0, nsamples)

  # Now, try to fit a spline to the accumulated accuracy
  nknots = splines
  kx = np.linspace (0.0, 1.0, nknots)
  spline = utils.Spline (percentile, integrated_accuracy - integrated_scores, kx, runout=spline_method)

  # Evaluate the spline to get the accuracy
  acc = spline.evaluate_deriv (percentile)
  acc += scores

  # Return the interpolating function -- uses full (not decimated) scores and
  # accuracy
  func = interpolated_function (scores, acc)
  return func

#------------------------------------------------------------------------------

def get_nth_results (scores, labels, n) :

  tscores = np.array([score[n] for score in scores])
  tacc = np.array([1.0 if n == label else 0.0 for label in labels])
  return tscores, tacc

#------------------------------------------------------------------------------


def get_top_results (scores, labels, nn, inclusive=False, return_topn_classid=False) :

  # Different if we want to take inclusing scores
  if inclusive : return get_top_results_inclusive (scores, labels, nn=nn)

  #  nn should be negative, -1 means top, -2 means second top, etc
  # Get the position of the n-th largest value in each row
  topn = [np.argpartition(score, nn)[nn] for score in scores]
  nthscore = [score[n] for score, n in zip (scores, topn)]
  labs = [1.0 if int(label) == int(n) else 0.0 for label, n in zip(labels, topn)]

  # Change to tensor
  tscores = np.array (nthscore)
  tacc = np.array(labs)

  if return_topn_classid:
    return tscores, tacc, topn
  else:
    return tscores, tacc

#------------------------------------------------------------------------------


def get_top_results_inclusive (scores, labels, nn=-1) :
  #  nn should be negative, -1 means top, -2 means second top, etc
  # Order scores in each row, so that nn-th score is in nn-th place
  order = np.argpartition(scores, nn)

  # Reorder the scores accordingly
  top_scores = np.take_along_axis (scores, order, axis=-1)[:,nn:]

  # Get the top nn lables
  top_labels = order[:,nn:]

  # Sum the top scores
  sumscores = np.sum(top_scores, axis=-1)

  # See if label is in the top nn
  labs = np.array([1.0 if int(label) in n else 0.0 for label, n in zip(labels, top_labels)])

  return sumscores, labs

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

import torch
class _MCELoss():
  def __init__(self, n_bins=15):
    super(_MCELoss, self).__init__()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]

  def eval(self, confidences, accuracies):
    mce = np.zeros(1)
    max_ce = 0.0
    for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
        # Calculated |confidence - accuracy| in each bin
      in_bin = confidences.__gt__(bin_lower) * confidences.__le__(bin_upper)
      prop_in_bin = in_bin.astype(float).mean()
      if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].astype(float).mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        if max_ce < np.abs(avg_confidence_in_bin - accuracy_in_bin):
          max_ce = np.abs(avg_confidence_in_bin - accuracy_in_bin)
    mce[0] = max_ce
    return mce


def MCE(logits, labels, n_bins=25):
  ece_criterion = _MCELoss(n_bins)
  confidences = torch.from_numpy(logits).cuda()
  accuracies = torch.from_numpy(labels).cuda()

  acc = accuracies.float().sum() / len(accuracies)
  ece = ece_criterion.eval(confidences.cpu().numpy(), accuracies.cpu().numpy())
  return acc, ece

#------------------------------------------------------------------------------


def cal_splines(y_probs_val, y_val, y_probs_test, y_test, ece_criterion, spline_method='natural',
                       splines=6):

  # for top-class calibration error
  n=-1


  scores1, labels1, scores1_class = get_top_results (y_probs_val, y_val, n, return_topn_classid=True)
  scores2, labels2, scores2_class = get_top_results (y_probs_test, y_test, n, return_topn_classid=True)

  y_probs_binary_val = np.zeros((y_probs_val.shape[0], 2))
  y_probs_binary_test = np.zeros((y_probs_test.shape[0], 2))
  y_probs_binary_val[np.arange(scores1.shape[0]), 0] = scores1
  y_probs_binary_test[np.arange(scores2.shape[0]), 0] = scores2
  y_probs_binary_val[np.arange(scores1.shape[0]), 1] = 1.0-scores1
  y_probs_binary_test[np.arange(scores2.shape[0]), 1] = 1.0-scores2

  y_val_binary_onehot = np.zeros((y_probs_val.shape[0], 2))
  y_test_binary_onehot = np.zeros((y_probs_test.shape[0], 2))
  y_val_binary_onehot[:, 0] = labels1
  y_test_binary_onehot[:, 0] = labels2
  y_val_binary_onehot[:, 1] = 1-labels1
  y_test_binary_onehot[:, 1] = 1-labels2

  # Plot the first set
  # KSElinf_uncal_val = plot_KS_graphs (scores1, labels1, spline_method, splines)
  # KSElinf_uncal_test = plot_KS_graphs (scores2, labels2, spline_method, splines)
  ece_uncal_val = ece_criterion.eval(scores1, labels1)
  ece_uncal_test = ece_criterion.eval(scores2, labels2)

  # Get recalibration function, based on scores1
  frecal = get_recalibration_function (scores1, labels1, spline_method, splines)

  # Recalibrate scores1 and plot
  scores1 = np.array([frecal(float(sc)) for sc in scores1])
  scores1[scores1 < 0.0] = 0.0
  scores1[scores1 > 1.0] = 1.0
  # KSElinf_cal_val = plot_KS_graphs (scores1, labels1, spline_method, splines)

  # Recalibrate scores2 and plot
  scores2 = np.array([frecal(float(sc)) for sc in scores2])
  scores2[scores2 < 0.0] = 0.0
  scores2[scores2 > 1.0] = 1.0
  # KSElinf_cal_test = plot_KS_graphs (scores2, labels2, spline_method, splines)
  ece_cal_val = ece_criterion.eval(scores1, labels1)
  ece_cal_test = ece_criterion.eval(scores2, labels2)
  print('Val ECE: %4f , Test ECE: %4f' %(ece_cal_val, ece_cal_test))
  
  acc, mce_test = MCE(scores2, labels2)
  print('ACC: %4f , MCE: %4f' %(acc, mce_test))

  return scores2, labels2
  # with open('saved_logits/cal_resnet152_imgnet_logits.p', 'wb') as f:
  #   pickle.dump([(scores2, labels2)], f)

  # # for top 2nd-class calibration error
  # n=-2
  # newtitle_val = title_val + f" Class[{n}]"
  # newtitle_test = title_test + f" Class[{n}]"
  #
  # scores1, labels1   = get_top_results (y_probs_val, y_val, n)
  # scores2, labels2   = get_top_results (y_probs_test, y_test, n)
  #
  # # Plot the first set
  # KSE2linf_uncal_val = plot_KS_graphs (scores1, labels1, spline_method, splines, outdir, 'uncalibrated_val_class'+str(n), title=newtitle_val+" | Uncalibrated")
  # KSE2linf_uncal_test = plot_KS_graphs (scores2, labels2, spline_method, splines, outdir, 'uncalibrated_test_class'+str(n), title=newtitle_test+" | Uncalibrated")
  #
  # # Get recalibration function, based on scores1
  # frecal = get_recalibration_function (scores1, labels1, spline_method, splines)
  #
  # # Recalibrate scores1 and plot
  # scores1 = np.array([frecal(float(sc)) for sc in scores1])
  # scores1[scores1 < 0.0] = 0.0
  # scores1[scores1 > 1.0] = 1.0
  # KSE2linf_cal_val = plot_KS_graphs (scores1, labels1, spline_method, splines, outdir, 'spline_calibrated_val_class'+str(n), title=newtitle_val+" | Calibrated")
  #
  # # Recalibrate scores2 and plot
  # scores2 = np.array([frecal(float(sc)) for sc in scores2])
  # scores2[scores2 < 0.0] = 0.0
  # scores2[scores2 > 1.0] = 1.0
  # KSE2linf_cal_test = plot_KS_graphs (scores2, labels2, spline_method, splines, outdir, 'spline_calibrated_test_class'+str(n), title=newtitle_test+" | Calibrated")


  ###################################################
  ### Estimating Within-k class KS score
  # n=-2
  # newtitle_val = title_val + f" Class[{n}]"
  # newtitle_test = title_test + f" Class[{n}]"
  #
  # scores1, labels1   = get_top_results (y_probs_val, y_val, n, inclusive=True)
  # scores2, labels2   = get_top_results (y_probs_test, y_test, n, inclusive=True)
  #
  # # Plot the first set
  # KSE_wn_2linf_uncal_val = plot_KS_graphs (scores1, labels1, spline_method, splines, outdir, 'uncalibrated_val_class_wn'+str(n), title=newtitle_val+" | Uncalibrated")
  # KSE_wn_2linf_uncal_test = plot_KS_graphs (scores2, labels2, spline_method, splines, outdir, 'uncalibrated_test_class_wn'+str(n), title=newtitle_test+" | Uncalibrated")
  #
  # # Get recalibration function, based on scores1
  # frecal = get_recalibration_function (scores1, labels1, spline_method, splines)
  #
  # # Recalibrate scores1 and plot
  # scores1 = np.array([frecal(float(sc)) for sc in scores1])
  # scores1[scores1 < 0.0] = 0.0
  # scores1[scores1 > 1.0] = 1.0
  # KSE_wn_2linf_cal_val = plot_KS_graphs (scores1, labels1, spline_method, splines, outdir, 'spline_calibrated_val_class_wn'+str(n), title=newtitle_val+" | Calibrated")
  #
  # # Recalibrate scores2 and plot
  # scores2 = np.array([frecal(float(sc)) for sc in scores2])
  # scores2[scores2 < 0.0] = 0.0
  # scores2[scores2 > 1.0] = 1.0
  # KSE_wn_2linf_cal_test = plot_KS_graphs (scores2, labels2, spline_method, splines, outdir, 'spline_calibrated_test_class_wn'+str(n), title=newtitle_test+" | Calibrated")
  #
  # results_beforecalib.add(
  #     val_PECE=ece_uncal_val[0], test_PECE=ece_uncal_test[0],
  #     val_PKSE_linf=KSElinf_uncal_val, test_PKSE_linf=KSElinf_uncal_test,
  #     val_KSE2_linf=KSE2linf_uncal_val, test_KSE2_linf=KSE2linf_uncal_test,
  #     val_KSE_wn_2_linf=KSE_wn_2linf_uncal_val, test_KSE_wn_2_linf=KSE_wn_2linf_uncal_test,
  # )
  # results_beforecalib.save()
  #
  # results_aftercalib.add(
  #     val_PECE=ece_cal_val[0], test_PECE=ece_cal_test[0],
  #     val_PKSE_linf=KSElinf_cal_val, test_PKSE_linf=KSElinf_cal_test,
  #     val_KSE2_linf=KSE2linf_cal_val, test_KSE2_linf=KSE2linf_cal_test,
  #     val_KSE_wn_2_linf=KSE_wn_2linf_cal_val, test_KSE_wn_2_linf=KSE_wn_2linf_cal_test,
  # )
  # results_aftercalib.save()

  # return ece_uncal_val[0], ece_uncal_test[0], KSElinf_uncal_val, KSElinf_uncal_test, KSE2linf_uncal_val, KSE2linf_uncal_test, \
  #        KSE_wn_2linf_uncal_val, KSE_wn_2linf_uncal_test, \
  #        ece_cal_val[0], ece_cal_test[0], \
  #        KSElinf_cal_val, KSElinf_cal_test, KSE2linf_cal_val, KSE2linf_cal_test, \
  #        KSE_wn_2linf_cal_val, KSE_wn_2linf_cal_test

#------------------------------------------------------------------------------

def first_above (A, val, low=0, high=-1):
  # Find the first time that the array exceeds, or equals val in the range low to high
  # inclusive -- this uses binary search

  # Initialization
  if high == -1: high = utils.len0(A)-1

  # Stopping point, when interval reduces to one element
  if high == low:
    if val <= A[low]:
      return low
    else :
      # The element does not exist.  This means that there is nowhere
      # in the array where A[k] >= val
      return low+1    # This will be out-of-bounds if the array never exceeds val

  # Otherwise, we subdivide and continue -- mid must be less then high
  # but can equal low, when high-low = 1
  mid = low + (high - low) // 2

  if A[mid] >= val:
    # In this case, the first time must be in the interval [low, mid]
    return first_above (A, val, low, mid)
  else :
    # In this case, the first time A[k] exceeds val must be to the right
    return first_above (A, val, mid+1, high)

if __name__ == '__main__' :
    ece_criterion = _ECELoss(n_bins=25)
    fname = '/home/ycliu/zouyl/uncertainty/logits/probs_lenet5_c10_logits.p'
    ((y_probs_val, y_val), (y_probs_test, y_test)) = unpickle_probs(fname)

    confidences_test, labels_test = cal_splines(y_probs_val, y_val, y_probs_test, y_test, ece_criterion,)

    print(confidences_test[:10])
    print(confidences_test.shape)
    print(labels_test.shape)