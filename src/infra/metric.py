import csv
import os

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score


def per_segment(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = cfm[0, 0], cfm[1, 1], cfm[1, 0], cfm[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1 = f1_score(y_true, y_pred, average='binary')
    return cfm, acc, sn, sp, f1
