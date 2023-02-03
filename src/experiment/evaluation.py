import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras

import src.infra.metric as metrics
from src.experiment.util import plot_and_save_cfm

def final_test(log_dir, x_test, x_test_5min, y_test, recording_name_test):
    # load trained model
    weights_filepath = log_dir + '/checkpoint/classifier.weights.best.hdf5'
    custom_objects = {'tf': tf}
    model = keras.models.load_model(weights_filepath, custom_objects=custom_objects)

    # save prediction score
    y_score = model.predict([x_test, x_test_5min], batch_size=32, verbose=1)
    y_pred = np.argmax(y_score, axis=-1)
    # y_pred = model.predict([x_test, x_test_5min], batch_size=1024, verbose=1)
    # y_pred = np.int64(y_pred >= 0.5).flatten()

    # per segment performance
    cfm, acc, sn, sp, f1 = metrics.per_segment(y_test, y_pred)
    print("acc: {}, sn: {}, sp: {}, f1: {}".format(acc * 100, sn * 100, sp * 100, f1))
    label_names = ['SA', 'NSA']
    # cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    plot_and_save_cfm(log_dir, cfm, 'cfm-per-segment', label_names)
    # per recording performance
    with open("../../resources/apnea-ecg-database-1.0.0/additional-information.txt", "r") as f:
        original = []
        for line in f:
            rows = line.strip().split("\t")
            if len(rows) == 12:
                if rows[0].startswith("x"):
                    original.append([rows[0], float(rows[3]) / float(rows[1]) * 60])

    original_AHI = pd.DataFrame(original, columns=["subject", "true_AHI"])
    original_AHI = original_AHI.set_index("subject")
    original_AHI.name = 'true_AHI'

    per_segment_pred = pd.DataFrame({"y_pred": y_pred, "subject": recording_name_test})
    predict_AHI = per_segment_pred.groupby(by="subject").apply(lambda d: d["y_pred"].mean() * 60)
    predict_AHI.name = 'predict_AHI'

    true_pred_AHI = pd.concat([original_AHI, predict_AHI], axis=1)

    corr = true_pred_AHI.corr()

    true_pred_AHI = true_pred_AHI.applymap(lambda a: int(a > 5))

    cfm = confusion_matrix(true_pred_AHI['true_AHI'], true_pred_AHI['predict_AHI'], labels=(1, 0))

    plot_and_save_cfm(log_dir, cfm, 'cfm-per-recording', label_names)

    TP, TN, FP, FN = cfm[0, 0], cfm[1, 1], cfm[1, 0], cfm[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    print("acc: {}, sn: {}, sp: {}, corr: {}".format(acc * 100, sn * 100, sp * 100, corr['true_AHI']['predict_AHI']))