import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from logger import LB_Logger


class Global_vars:
    config = {
        "randomness": 0.5,
        "direction_1": 0.0,
        "direction_2": 0.0,
        "direction_3": 0.0,
        "direction_4": 0.0,
        "direction_5": 0.0,
        "direction_6": 0.0,
    }


global_vars = Global_vars()
lb_logger = LB_Logger()


def load_config():
    '''
    Loads config from config.json.
    '''
    try:
        with open("config.json", "r") as f:
            global_vars.config = json.load(f)
        lb_logger.log_info("Config loaded.")
    except Exception as e:
        lb_logger.log_error("Failed to load config.json: " + str(e))


def save_config():
    '''
    Saves config to config.json.
    '''
    try:
        with open("config.json", "w") as f:
            json.dump(global_vars.config, f)
        lb_logger.log_info("Config saved.")
    except Exception as e:
        lb_logger.log_error("Failed to save config.json: " + str(e))


def import_train_data(filename: str):
    '''
    Imports train data from train_data.csv.
    '''
    try:
        train_data = pd.read_csv(filename)
        lb_logger.log_info("Train data imported.")
        # chop off the last column and save it as labels
        train_labels = train_data.pop('result')
        # convert every label to a one-hot vector
        train_labels = tf.one_hot(train_labels, depth=6)
        return train_data, train_labels
    except Exception as e:
        lb_logger.log_error("Failed to import train data: " + str(e))


if __name__ == "__main__":
    train_data, train_labels = import_train_data("train_data.csv")

    model = keras.Sequential([
        # keras.layers.Dense(6, activation='relu'),
        keras.layers.Dense(6, activation='sigmoid'),
        # keras.layers.Dense(6)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(train_data,  train_labels, verbose=2)

    lb_logger.log_info('Test accuracy: {}'.format(test_acc))
    lb_logger.log_info('Test loss: {}'.format(test_loss))

    test_pd0 = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=train_data.columns)
    prediction = model.predict(test_pd0)[0]
    print('0: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
          str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    test_pd1 = pd.DataFrame([[1, 0, 0, 0, 0, 0]], columns=train_data.columns)
    prediction = model.predict(test_pd1)[0]
    print('1: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
          str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    test_pd2 = pd.DataFrame([[0, 1, 0, 0, 0, 0]], columns=train_data.columns)
    prediction = model.predict(test_pd2)[0]
    print('2: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
          str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    test_pd3 = pd.DataFrame([[0, 0, 1, 0, 0, 0]], columns=train_data.columns)
    prediction = model.predict(test_pd3)[0]
    print('3: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
          str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    test_pd4 = pd.DataFrame([[0, 0, 0, 1, 0, 0]], columns=train_data.columns)
    prediction = model.predict(test_pd4)[0]
    print('4: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
          str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    test_pd5 = pd.DataFrame([[0, 0, 0, 0, 1, 0]], columns=train_data.columns)
    prediction = model.predict(test_pd5)[0]
    print('5: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
          str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    test_pd6 = pd.DataFrame([[0, 0, 0, 0, 0, 1]], columns=train_data.columns)
    prediction = model.predict(test_pd6)[0]
    print('6: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
          str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))
