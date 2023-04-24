import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from math import ceil
import wfdb

from pathlib import Path
import sys
srcpath = os.path.abspath('../../')
sys.path.append(srcpath)
from src.infra.preprocessing import preprocess_recording

data_dir = Path('../../resources/kangwon_hospital/')

# number of threads to preprocess recording
threads = os.cpu_count()
threads = 61 # maximum num_workers on Windows
released_set_recording_names = ['0001']
# released_set_recording_names = [
#     name.split('.')[0] for name in os.listdir(data_dir)
#     if name.endswith('.npy')
# ]
released_set_recording_names.sort()

random.seed(42)
training_set_recording_names = random.sample(released_set_recording_names, int(0.9*len(released_set_recording_names)))
test_set_recording_names = [name for name in released_set_recording_names if name not in training_set_recording_names]

# for test:
# training_set_recording_names = ['0001']
# test_set_recording_names = ['0001']

def preprocess_released_set(released_set_recording_names):
    print('start: preprocess released set')

    o_train, o_train_5, y_train, groups_train = [], [], [], []
    
    for i in range(threads):
        with ProcessPoolExecutor(max_workers=threads) as executor:
            task_list = []
            for recording_name in released_set_recording_names[i*ceil(len(released_set_recording_names)/threads):
                                                               (i+1)*ceil(len(released_set_recording_names)/threads)]:

                # Seola, 0414, for Kangwon hospital dataset
                with open(data_dir/(recording_name+'.pkl'), 'rb') as f:
                    labels = pickle.load(f)
                signal_recording = np.load(data_dir/(recording_name+'.npy'))
                
                # preprocess_recording(signal_recording, 100, labels, recording_name)
                # sampling rate = 100 Hz (otherwise resampled to 100 Hz)
                task_list.append(executor.submit(preprocess_recording, signal_recording, 100, labels, recording_name))

            for task in as_completed(task_list):
                X, X_with_adjacent_segment, y, groups = task.result()
                o_train.extend(X)
                o_train_5.extend(X_with_adjacent_segment)
                y_train.extend(y)
                groups_train.extend(groups)

            if i in [0,1,2]:
                print(len(o_train))    

    o_train = np.array(o_train, dtype="float32")
    y_train = np.array(y_train, dtype="float32")
    print('end: preprocess released set\n')

    return o_train, o_train_5, y_train, groups_train


def preprocess_apnea_ecg_database():
    """
        preprocess Kangwon Univ ECG data
    """
    # if os.path.exists('../../output/preprocessed/kangwon-apnea-ecg.pkl'):
    #     return

    if os.path.exists('../../output/preprocessed/apnea-ecg-physionet.pkl'):
        print('yes')
        return

    o_train, o_train_5, y_train, groups_train = preprocess_released_set(training_set_recording_names)
    o_test, o_test_5, y_test, groups_test = preprocess_released_set(test_set_recording_names)

    apnea_ecg = dict(o_train=o_train, o_train_5=o_train_5, y_train=y_train, groups_train=groups_train,
                     o_test=o_test, o_test_5=o_test_5, y_test=y_test,
                     groups_test=groups_test)
    with open('../../output/preprocessed/kangwon-apnea-ecg.pkl', 'wb') as f:
        pickle.dump(apnea_ecg, f, protocol=2)

    print('preprocess kangwon-univ-hospital ECG finish!')
    print('preprocessed file saved as \'output/preprocessed/kangwon-apnea-ecg.pkl\' \n')


def load_apnea_ecg_database_preprocessed_data():
    print('start: loading Apnea-ECG preprocessed data')

    # with open('../../output/preprocessed/kangwon-apnea-ecg.pkl', 'rb') as f:
    #     apnea_ecg = pickle.load(f)

    with open('../../output/preprocessed/apnea-ecg-physionet.pkl', 'rb') as f:
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