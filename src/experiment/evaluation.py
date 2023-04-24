import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow import keras

import sys
sys.path.append('C:/Users/user/Desktop/Sleep_Apnea/DSF/DSF-SANet/')
import src.infra.metric as metrics
from src.experiment.util import plot_and_save_cfm

def final_test(log_dir, x_test, x_test_5min, y_test, recording_name_test):
    # load trained model
    weights_filepath = log_dir + '/checkpoint/classifier.weights.best.hdf5'
    custom_objects = {'tf': tf}
    model = keras.models.load_model(weights_filepath, custom_objects=custom_objects)

    # save prediction score
    y_pred = model.predict([x_test, x_test_5min], batch_size=1024, verbose=1)
    y_pred = np.int64(y_pred >= 0.5).flatten()

    # per segment performance
    cfm, acc, sn, sp, f1, auc = metrics.per_segment(y_test, y_pred)
    print("acc: {}, sn: {}, sp: {}, f1: {}, auc: {}".format(acc * 100, sn * 100, sp * 100, f1, auc))
    label_names = ['SA', 'NSA']
    cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    plot_and_save_cfm(log_dir, cfm, 'cfm-per-segment', label_names)