import numpy as np
import matplotlib.pyplot as plt
import os
from .. import splines_utils as utils


def ensure_numpy(a):
    if not isinstance(a, np.ndarray): a = a.numpy()
    return a


def compute_accuracy (scores_in, labels_in, spline_method, splines) :

    # Computes the accuracy given scores and labels.
    # Also plots a graph of the spline fit

    # Change to numpy, then this will work
    scores = ensure_numpy (scores_in)
    labels = ensure_numpy (labels_in)

    # Sort them
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]

    #Accumulate and normalize by dividing by num samples
    nsamples = utils.len0(scores)
    integrated_accuracy = np.cumsum(labels) / nsamples
    integrated_scores   = np.cumsum(scores) / nsamples
    percentile = np.linspace (0.0, 1.0, nsamples)

    # Now, try to fit a spline to the accumulated accuracy
    nknots = splines
    kx = np.linspace (0.0, 1.0, nknots)

    error = integrated_accuracy - integrated_scores
    #error = integrated_accuracy

    spline = utils.Spline (percentile, error, kx, runout=spline_method)

    # Now, compute the accuracy at the original points
    dacc = spline.evaluate_deriv (percentile)
    #acc = dacc
    acc = scores + dacc

    # Compute the error
    fitted_error = spline.evaluate (percentile)
    err = error - fitted_error
    stdev = np.sqrt(np.mean(err*err))
    print (f"compute_error: fitted spline with accuracy {utils.str(stdev, form='{:.3e}')}")

    return acc, -fitted_error


def plot_KS_graphs(scores, labels, spline_method, splines):
    # KS stands for Kolmogorov-Smirnov
    # Plots a graph of the scores and accuracy
    tks = utils.Timer ("Plotting graphs")

    # Change to numpy, then this will work
    scores = ensure_numpy (scores)
    labels = ensure_numpy (labels)

    # Sort the data
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = utils.len0(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy   = np.cumsum(labels) / nsamples
    percentile = np.linspace (0.0, 1.0, nsamples)
    fitted_accuracy, fitted_error = compute_accuracy (scores, labels, spline_method, splines)

    # Work out the Kolmogorov-Smirnov error
    KS_error_max = np.amax(np.absolute (integrated_scores - integrated_accuracy))

    return KS_error_max
