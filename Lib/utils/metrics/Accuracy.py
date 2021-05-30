import numpy as np
from sklearn import metrics
import torch
from Lib.utils.generic.generic_utils import mean


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
"""
Author: Alessandro Cattoi
Description: This file implements a class who calculates the accuracy of the classification results
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Accuracy:
    """
    NB always use at least batch of 16 because otherwise accuracy_w is very inaccurate.
    If it calculates accuracy and a class is not present any false positive of that class is not counted so the accuracy is
    strongly overestimated
    This class get prediction and label tensor of any shape it flat them removes pixel that has not a label and calculate an
    overall accuracy of the tensor. Each value calculated is then stored in a list.
    At any time using the get mean dict is possible to return the mean value of the lists of scores store in a dictionary.
        Labels are composed as follow:
            CLASS NAME            VALUE
        - forest            -->     0
        - street            -->     1
        - field             -->     2
        - urban             -->     3
        - water             -->     4
        - Not classified    -->    255
    """
    def __init__(self, labels=[0, 1, 2, 3, 4]):
        super(Accuracy, self).__init__()
        self.f1 = []
        self.AA = []
        self.weighted_acc = []
        self.support = []
        self.labels = labels

    def update_acc(self, y_true, y_pred):
        """
        Calculates accuracy scores for classification
        AA is all good pixel divided by all pixel
        weighted_acc is the weighted mean between each class correct pixel over total pixel classified per class
        :param y_true: labels
        :param y_pred: prediction
        :return: accuracy dict with keys: f1, AA, AA_w
        """
        y_true = y_true.cpu().numpy()
        y_pred = torch.argmax(y_pred, 1)
        y_pred = y_pred.cpu().numpy()

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        index_f = np.argwhere(y_true_flat == 255)
        y_true_flat = np.delete(y_true_flat, index_f)
        y_pred_flat = np.delete(y_pred_flat, index_f)
        # print(metrics.classification_report(y_true_flat, y_pred_flat, zero_division=0))
        # print(metrics.confusion_matrix(y_true_flat, y_pred_flat))

        self.f1.append(metrics.f1_score(y_true_flat, y_pred_flat, labels=self.labels, average='macro', zero_division=0)*len(y_true_flat))
        self.AA.append(metrics.accuracy_score(y_true_flat, y_pred_flat)*len(y_true_flat))
        self.support.append(len(y_true_flat))
        self.weighted_acc.append(metrics.precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)*len(y_true_flat))

    def get_mean_dict(self):
        """
        Calculates mean stored in the calss list attributes and return them in a dictionary
         (that can easily stored in a Logger or loaded in tensorboard)
        :return:
        """
        accuracy = {
            'f1': mean(self.f1, self.support),
            'AA': mean(self.AA, self.support),
            'AA_w': mean(self.weighted_acc, self.support),
        }
        return accuracy

    def reinit(self):
        self.f1 = []
        self.AA = []
        self.weighted_acc = []
        self.support = []
