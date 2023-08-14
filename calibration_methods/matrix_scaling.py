import torch
# from pytorchtools import EarlyStopping
import numpy as np

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


class MatrixScaling(torch.nn.Module):

    def __init__(self, classes=-1, max_epochs=5000, patience=7):
        """
        Initialize class

        Params:
            max_epochs (int): maximum iterations done by optimizer.
            classes (int): how many classes in given data set. (based on logits )
            patience (int): how many worse epochs before early stopping
        """
        super(MatrixScaling, self).__init__()
        if classes >= 1:
            self.model = self.create_model(classes)
        else:
            self.model = None
        self.max_epochs = max_epochs
        self.patience = patience
        self.classes = classes

    def create_model(self, classes):
        model = torch.nn.Sequential(
            torch.nn.Linear(classes,classes,bias=True),
            # torch.nn.Softmax(dim=-1),
        )
        return model

    # Find the temperature
    def fit(self, logits, labels):
        """
        Trains the model and finds optimal parameters

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the model after minimizing is finished.
        """
        # cifar10 : lr=0.01 step_size=200
        self.cuda()
        nll_criterion = torch.nn.CrossEntropyLoss().cuda()
        # ece_criterion = _ECELoss().cuda()
        early_stop = EarlyStopping(self.patience, verbose=True, delta=0.0001)
        # optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.01, max_iter=70)
        # def eval():
        #     optimizer.zero_grad()
        #     loss = nll_criterion(self.model(logits), labels)
        #     loss.backward()
        #     # early_stop(loss)
        #     # if early_stop.early_stop:
        #     #     print('early stop {} loss: {}'.format(i, early_stop.best_score))
        #     #     return 0
        #     return loss
        # optimizer.step(eval)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1)
        for i in range(self.max_epochs):
            optimizer.zero_grad()
            loss = nll_criterion(self.model(logits), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            early_stop(loss)
            # print("epoch: {} , loss : {}".format(i, loss))
            if early_stop.early_stop:
                print('early stop {} loss: {}'.format(i, early_stop.best_score))
                break

        return self

    def forward(self, x):
        return self.model(x)

import logging
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
from .dirichlet_utils.multinomial import MultinomialRegression
from .dirichlet_utils.utils import clip_for_log
from sklearn.metrics import log_loss

from .dirichlet_utils.multinomial import _get_identity_weights


class MatrixScaling_SK(BaseEstimator, RegressorMixin):
    def __init__(self, reg_lambda_list=[0.0], reg_mu_list=[None],
                 logit_input=False, logit_constant=None,
                 weights_init=None, initializer='identity'):
        self.weights_init = weights_init
        self.logit_input = logit_input
        self.logit_constant = logit_constant
        self.reg_lambda_list = reg_lambda_list
        self.reg_mu_list = reg_mu_list
        self.initializer = initializer

    def __setup(self):
        self.reg_lambda = 0.0
        self.reg_mu = None
        self.calibrator_ = None
        self.weights_ = self.weights_init

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        self.__setup()

        k = np.shape(X)[1]

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        if self.logit_input == False:
            _X = np.copy(X)
            _X = np.log(clip_for_log(_X))
            _X_val = np.copy(X_val)
            _X_val = np.log(clip_for_log(X_val))
            if self.logit_constant is None:
                _X = _X - _X[:, -1].reshape(-1, 1).repeat(k, axis=1)
                _X_val = _X_val[:, -1].reshape(-1, 1).repeat(k, axis=1)
            else:
                _X = _X - self.logit_constant
                _X_val = _X_val - self.logit_constant
        else:
            _X = np.copy(X)
            _X_val = np.copy(X_val)

        for i in range(0, len(self.reg_lambda_list)):
            for j in range(0, len(self.reg_mu_list)):
                tmp_cal = MultinomialRegression(method='Full',
                                                reg_lambda=self.reg_lambda_list[i],
                                                reg_mu=self.reg_mu_list[j])
                tmp_cal.fit(_X, y, *args, **kwargs)
                tmp_loss = log_loss(y_val, tmp_cal.predict_proba(_X_val))

                if (i + j) == 0:
                    final_cal = tmp_cal
                    final_loss = tmp_loss
                    final_reg_lambda = self.reg_lambda_list[i]
                    final_reg_mu = self.reg_mu_list[j]
                elif tmp_loss < final_loss:
                    final_cal = tmp_cal
                    final_loss = tmp_loss
                    final_reg_lambda = self.reg_lambda_list[i]
                    final_reg_mu = self.reg_mu_list[j]

        self.calibrator_ = final_cal
        self.reg_lambda = final_reg_lambda
        self.reg_mu = final_reg_mu
        self.weights_ = self.calibrator_.weights_

        return self

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        k = np.shape(S)[1]

        if self.logit_input == False:
            _S = np.log(clip_for_log(np.copy(S)))
            if self.logit_constant is None:
                _S = _S - _S[:, -1].reshape(-1, 1).repeat(k, axis=1)
            else:
                _S = _S - self.logit_constant
        else:
            _S = np.copy(S)

        return np.asarray(self.calibrator_.predict_proba(_S))

    def predict(self, S):
        k = np.shape(S)[1]

        if self.logit_input == False:
            _S = np.log(clip_for_log(np.copy(S)))
            if self.logit_constant is None:
                _S = _S - _S[:, -1].reshape(-1, 1).repeat(k, axis=1)
            else:
                _S = _S - self.logit_constant
        else:
            _S = np.copy(S)

        return np.asarray(self.calibrator_.predict(_S))