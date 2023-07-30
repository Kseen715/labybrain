from logger import LB_Logger
from tqdm.keras import TqdmCallback
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Global_vars:
    config = {
        "randomness": 0.5,
        "dir_1": 0.0,
        "dir_2": 0.0,
        "dir_3": 0.0,
        "dir_4": 0.0,
        "dir_5": 0.0,
        "dir_6": 0.0,
    }
    model = keras.Sequential()
    model_filename = ""


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
        train_labels = train_labels - 1
        train_labels = tf.one_hot(train_labels, depth=6)
        return train_data, train_labels
    except Exception as e:
        lb_logger.log_error("Failed to import train data: " + str(e))


def load_model(model_filename: str):
    '''
    Loads the model from model.keras.
    '''
    try:
        global_vars.model = keras.models.load_model(model_filename)
        lb_logger.log_info("Model " + model_filename + " loaded.")
        return global_vars.model
    except Exception as e:
        lb_logger.log_error("Failed to load model: " + str(e))


def predict(predict_pd: pd.DataFrame):
    '''
    Predicts the result based on config.
    '''
    prediction = global_vars.model.predict(predict_pd)[0]
    max_prediction = max(prediction)
    # prediction_id = np.where(prediction, max(prediction))[0] + 1
    prediction_id = np.where(np.isclose(prediction, max(prediction)))[0] + 1
    # if prediction_id is list:
    prediction_id = prediction_id[0]
    return prediction, max_prediction, prediction_id
    # ('MDL: ' + global_vars.model_filename + '\n' +
    #         'INPT:\n' +
    #         str(predict_pd) + '\n' +
    #         'PRDT: ' + str(prediction) + '\n' +
    #         'RSLT: ' + str(max_prediction) + ' == ' +
    #         str(prediction_id))


def insert_randomity(randomness: float, predict_pd: pd.DataFrame):
    '''
    Inserts randomity into config data.
    '''
    local_pr_pd = predict_pd.copy()
    # get random number from 0 to 1

    # add random number to every dir, cap it at 0 and 1
    for i in range(1, 7):
        random_number = np.random.rand()
        sign_of_random_number = np.random.choice([-1, 1])
        random_number = sign_of_random_number * random_number * randomness
        local_pr_pd["dir_" + str(i)] = \
            np.clip(local_pr_pd["dir_" + str(i)] + random_number, 0, 1)
    return local_pr_pd


if __name__ == "__main__":
    train_data, train_labels = import_train_data("train_data.csv")
    # lb_logger.log_info("Train data: " + str(train_data))
    # lb_logger.log_info("Train labels: " + str(train_labels))

    model = keras.Sequential([
        keras.layers.Dense(6, activation='relu'),
        keras.layers.Dense(3, activation='relu'),
        keras.layers.Dense(6, activation='sigmoid'),
        # keras.layers.Dense(6)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=1000,
              verbose=0, callbacks=[TqdmCallback(verbose=1)])

    test_loss, test_acc = model.evaluate(train_data,  train_labels, verbose=0)

    lb_logger.log_info('Test accuracy: {}'.format(test_acc))
    lb_logger.log_info('Test loss: {}'.format(test_loss))

    # save model
    max_model_id = 0
    for filename in os.listdir('models'):
        if filename.startswith('model_') and filename.endswith('.keras'):
            model_id = int(filename[6:-6])
            if model_id > max_model_id:
                max_model_id = model_id
    model_id = max_model_id + 1
    four_digit_model_id = str(model_id).zfill(4)
    model_filename = 'models/model_' + four_digit_model_id + '.keras'
    model.save(model_filename)
    lb_logger.log_info('Model saved as ' + model_filename)

    # print(predict())
    # test_pd0 = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=train_data.columns)
    # prediction = model.predict(test_pd0)[0]
    # print('0: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
    #       str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    # test_pd1 = pd.DataFrame([[1, 0, 0, 0, 0, 0]], columns=train_data.columns)
    # prediction = model.predict(test_pd1)[0]
    # print('1: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
    #       str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    # test_pd2 = pd.DataFrame([[0, 1, 0, 0, 0, 0]], columns=train_data.columns)
    # prediction = model.predict(test_pd2)[0]
    # print('2: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
    #       str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    # test_pd3 = pd.DataFrame([[0, 0, 1, 0, 0, 0]], columns=train_data.columns)
    # prediction = model.predict(test_pd3)[0]
    # print('3: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
    #       str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    # test_pd4 = pd.DataFrame([[0, 0, 0, 1, 0, 0]], columns=train_data.columns)
    # prediction = model.predict(test_pd4)[0]
    # print('4: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
    #       str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    # test_pd5 = pd.DataFrame([[0, 0, 0, 0, 1, 0]], columns=train_data.columns)
    # prediction = model.predict(test_pd5)[0]
    # print('5: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
    #       str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))

    # test_pd6 = pd.DataFrame([[0, 0, 0, 0, 0, 1]], columns=train_data.columns)
    # prediction = model.predict(test_pd6)[0]
    # print('6: ' + str(prediction) + '\t' + str(max(prediction)) + '\t' +
    #       str(np.where(np.isclose(prediction, max(prediction)))[0] + 1))
