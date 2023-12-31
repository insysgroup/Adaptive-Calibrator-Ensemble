import numpy as np

import torch
import torchvision as tv
import torchvision
import os
import ipdb
import tqdm
import time
import logging

# histogram binning
class HistogramBinning():
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.

    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """

    def __init__(self, M=15):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1. / M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1 + self.bin_size, self.bin_size)  # Set bin bounds for intervals

    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range

        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered) / nr_elems  # Sums positive classes
            return conf

    def fit(self, probs, true):
        """
        Fit the calibration model, finding optimal confidences for all the bins.

        Params:
            probs: probabilities of data
            true: true labels of data
        """

        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs=probs, true=true)
            conf.append(temp_conf)

        self.conf = conf

    # Fit based on predicted confidence
    def predict(self, probs):
        """
        Calibrate the confidences

        Param:
            probs: probabilities of the data (shape [samples, classes])

        Returns:
            Calibrated probabilities (shape [samples, classes])
        """

        # Go through all the probs and check what confidence is suitable for it.
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs
