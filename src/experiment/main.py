import numpy as np

from src.experiment.preprocess import preprocess_apnea_ecg_database, load_apnea_ecg_database_preprocessed_data
from src.experiment.evaluation import final_test
from src.experiment.training import train_classifier
from src.experiment.util import make_log_dir
from src.experiment.util import setup_gpu
from src.visualization.vis import vis_feature

selected_gpu_devices = '1'


def func():
    preprocess_apnea_ecg_database()
    x_train, x_train_5min, y_train, x_val, x_val_5min, y_val, x_test, x_test_5min, y_test, groups_test = load_apnea_ecg_database_preprocessed_data()

    setup_gpu(selected_gpu_devices)
    log_dir = make_log_dir()

    train_classifier(log_dir, x_train, x_train_5min, y_train, x_val, x_val_5min, y_val)

    final_test(log_dir, x_test, x_test_5min, y_test, groups_test)

if __name__ == '__main__':
    func()
