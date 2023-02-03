import datetime
import itertools
import os

import matplotlib.pyplot as plt
import tensorflow as tf


def setup_gpu(selected_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_devices
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_devices = tf.config.list_physical_devices(device_type='GPU')
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


def make_log_dir():
    log_dir = '../../output/log/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(log_dir + '/checkpoint')
    os.makedirs(log_dir + '/performance')
    os.makedirs(log_dir + '/predict')
    return log_dir


def plot_and_save_cfm(log_dir, cfm, title, label_names):
    num_classes = len(label_names)

    plt.imshow(cfm, cmap=plt.cm.Blues)
    # plt.title(title, size=16)
    # plt.xlabel("Prediction", size=24)
    # plt.ylabel("Ground truth", size=24)
    # plt.yticks(range(num_classes), label_names, size=24, weight='bold')
    # plt.xticks(range(num_classes), label_names, size=24, weight='bold')
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    for i, j in itertools.product(range(cfm.shape[0]), range(cfm.shape[1])):
        value = cfm[i, j]
        if value == 23 or value == 12 or value > 5000:
            color = 'white'
        else:
            color = 'black'
        plt.text(j, i, value, verticalalignment='center', horizontalalignment='center', size=40,
                 family="Times new roman", color=color, weight='bold')

    # plt.tight_layout()
    plt.subplots_adjust(left=1, bottom=1, right=2, top=2)

    # plt.colorbar()
    plt.savefig(log_dir + '/performance/' + title + '.png', bbox_inches='tight')
    plt.savefig(log_dir + '/performance/' + title + '.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
