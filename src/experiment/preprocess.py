import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import wfdb

import sys
srcpath = os.path.abspath('../../')
sys.path.append(srcpath)
from src.infra.preprocessing import preprocess_recording

# https://physionet.org/physiobank/database/apnea-ecg/
data_dir = '../../resources/apnea-ecg-database-1.0.0'
# number of threads to preprocess recording
num_worker = 35


def preprocess_released_set():
    print('start: preprocess released set')

    released_set_recording_names = [
        'a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10',
        'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20',
        'b01', 'b02', 'b03', 'b04', 'b05',
        'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10'
    ]

    o_train, o_train_5, y_train, groups_train = [], [], [], []

    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        task_list = []
        for i in range(len(released_set_recording_names)):
            recording_name = released_set_recording_names[i]
            labels = wfdb.rdann(os.path.join(data_dir, recording_name), extension='apn').symbol
            signal_recording = wfdb.rdrecord(os.path.join(data_dir, recording_name), channels=[0]).p_signal[:, 0]
            # preprocess_recording(signal_recording, 100, labels, recording_name)
            task_list.append(executor.submit(preprocess_recording, signal_recording, 100, labels, recording_name))

        for task in as_completed(task_list):
            X, X_with_adjacent_segment, y, groups = task.result()
            o_train.extend(X)
            o_train_5.extend(X_with_adjacent_segment)
            y_train.extend(y)
            groups_train.extend(groups)

    o_train = np.array(o_train, dtype="float32")
    y_train = np.array(y_train, dtype="float32")
    print('end: preprocess released set\n')

    return o_train, o_train_5, y_train, groups_train


def preprocess_withheld_set():
    print('start: preprocess withheld set')

    withheld_set_recording_names = [
        'x01', 'x02', 'x03', 'x04', 'x05', 'x06', 'x07', 'x08', 'x09', 'x10',
        'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
        'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30',
        'x31', 'x32', 'x33', 'x34', 'x35'
    ]
    answers = {}
    with open(os.path.join(data_dir, 'event-2-answers'), 'r') as f:
        for answer in f.read().split('\n\n'):
            answers[answer[:3]] = list(''.join(answer.split()[2::2]))

    o_test, o_test_5, y_test, groups_test = [], [], [], []

    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        task_list = []
        for i in range(len(withheld_set_recording_names)):
            recording_name = withheld_set_recording_names[i]
            labels = answers[recording_name]
            signal_recording = wfdb.rdrecord(os.path.join(data_dir, recording_name), channels=[0]).p_signal[:, 0]
            task_list.append(executor.submit(preprocess_recording, signal_recording, 100, labels, recording_name))

        for task in as_completed(task_list):
            X, X_with_adjacent_segment, y, groups = task.result()
            o_test.extend(X)
            o_test_5.extend(X_with_adjacent_segment)
            y_test.extend(y)
            groups_test.extend(groups)

    o_test = np.array(o_test, dtype="float32")
    y_test = np.array(y_test, dtype="float32")
    print('end: preprocess withheld set\n')

    return o_test, o_test_5, y_test, groups_test


def preprocess_apnea_ecg_database():
    """
        preprocess PhysioNet apnea-ecg-database-1.0.0
    """
    if os.path.exists('../../output/preprocessed/apnea-ecg.pkl'):
        return

    o_train, o_train_5, y_train, groups_train = preprocess_released_set()
    o_test, o_test_5, y_test, groups_test = preprocess_withheld_set()

    apnea_ecg = dict(o_train=o_train, o_train_5=o_train_5, y_train=y_train, groups_train=groups_train,
                     o_test=o_test, o_test_5=o_test_5, y_test=y_test,
                     groups_test=groups_test)
    with open('../../output/preprocessed/apnea-ecg.pkl', 'wb') as f:
        pickle.dump(apnea_ecg, f, protocol=2)

    print('preprocess PhysioNet apnea-ecg-database-1.0.0 finish!')
    print('preprocessed file saved as \'output/preprocessed/apnea-ecg.pkl\' \n')


def load_apnea_ecg_database_preprocessed_data():
    print('start: loading Apnea-ECG preprocessed data')

    with open('../../output/preprocessed/apnea-ecg.pkl', 'rb') as f:
        apnea_ecg = pickle.load(f)
    x_train, x_train_5min, y_train, groups_train = apnea_ecg["o_train"], apnea_ecg["o_train_5"], apnea_ecg["y_train"], \
                                                   apnea_ecg["groups_train"]
    x_test, x_test_5min, y_test, groups_test = apnea_ecg["o_test"], apnea_ecg["o_test_5"], apnea_ecg["y_test"], \
                                               apnea_ecg["groups_test"]

    print('end: loading Apnea-ECG preprocessed data\n')
    x_train = np.array(x_train, dtype="float32")
    x_train_5min = np.array(x_train_5min, dtype="float32")
    y_train = np.array(y_train, dtype="float32")
    x_test = np.array(x_test, dtype="float32")
    x_test_5min = np.array(x_test_5min, dtype="float32")
    y_test = np.array(y_test, dtype="float32")

    seed = 7
    random.seed(seed)
    idx_train = random.sample(range(len(x_train)), int(len(x_train) * 0.7))
    num = [i for i in range(len(x_train))]
    idx_val = set(num) - set(idx_train)
    idx_val = list(idx_val)

    x_training = x_train[idx_train]
    x_training_5min = x_train_5min[idx_train]
    y_training = y_train[idx_train]

    x_val = x_train[idx_val]
    x_val_5min = x_train_5min[idx_val]
    y_val = y_train[idx_val]

    return x_training, x_training_5min, y_training, x_val, x_val_5min, y_val, x_test, x_test_5min, y_test, groups_test
