import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import pandas as pd
import time
from sklearn.metrics import log_loss
from os.path import join
import sklearn.metrics as metrics
from os import path

# np.random.seed(333)

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))  # Subtract max so biggest is 0 to avoid numerical instability

    # Axis 0 if only one dimensional array
    axis = 0 if len(e_x.shape) == 1 else 1

    return e_x / e_x.sum(axis=axis, keepdims=1)

class VectorScaling():

    def __init__(self, classes=1, W=[], bias=[], maxiter=100, solver="BFGS", use_bias=True):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
            classes (int): how many classes in given data set. (based on logits )
            W (np.ndarray): matrix with temperatures for all the classes
            bias ( np.array): vector with biases
        """

        self.W = W
        self.bias = bias
        self.maxiter = maxiter
        self.solver = solver
        self.classes = classes
        self.use_bias = use_bias

    def _loss_fun(self, x, logits, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        W = np.diag(x[:self.classes])

        if self.use_bias:
            bias = x[self.classes:]
        else:
            bias = np.zeros(self.classes)
        scaled_probs = self.predict(logits, W, bias)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss

    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        true = true.flatten()  # Flatten y_val
        self.classes = logits.shape[1]
        x0 = np.concatenate([np.repeat(1, self.classes), np.repeat(0, self.classes)])
        opt = minimize(self._loss_fun, x0=x0, args=(logits, true), options={'maxiter': self.maxiter},
                       method=self.solver)
        self.W = np.diag(opt.x[:logits.shape[1]])
        self.bias = opt.x[logits.shape[1]:]

        return opt

    def predict(self, logits, W=[], bias=[]):
        """
        Scales logits based on the temperature and returns calibrated probabilities

        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.

        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if len(W) == 0 or len(bias) == 0:  # Use class variables
            scaled_logits = np.dot(logits, self.W) + self.bias
        else:  # Take variables W and bias from arguments
            scaled_logits = np.dot(logits, W) + bias

        return softmax(scaled_logits)

import torch

# torch.manual_seed(333)
# torch.cuda.manual_seed(333)
class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score+self.delta:
            self.counter+=1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
class VectorScaling_NN(torch.nn.Module):
    def __init__(self, classes=-1, max_epochs=5000, patience=15):
        super(VectorScaling_NN, self).__init__()
        self.weights = torch.nn.Parameter(torch.ones(classes) * 1.5)
        self.bias = torch.nn.Parameter(torch.zeros(classes))
        self.max_epochs = max_epochs
        self.patience = patience

    def forward(self, x):
        return torch.mul(x, self.weights) + self.bias

    # def pred(self, input, valid_indices):
    #     output = self.forward(input)
    #     if valid_indices:
    #         return output[:, valid_indices]
    #     else:
    #         return output

    def fit(self, logits, labels):
        self.cuda()
        nll_criterion = torch.nn.CrossEntropyLoss().cuda()
        # optimizer = torch.optim.LBFGS([self.weights, self.bias], lr=0.01, max_iter=50)
        early_stop = EarlyStopping(self.patience, verbose=True, delta=0.00001)
        # def eval():
        #     optimizer.zero_grad()
        #     loss = nll_criterion(self.forward(logits), labels)
        #     loss.backward()
        #     return loss
        # optimizer.step(eval)
        optimizer = torch.optim.Adam([self.weights, self.bias], lr=0.01)
        for i in range(self.max_epochs):
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            optimizer.step()
            early_stop(loss)
            # print(loss)
            if early_stop.early_stop:
                print('early stop {} loss: {}'.format(i, early_stop.best_score))
                break
        return self
