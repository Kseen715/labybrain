from logger import LB_Logger
from tqdm.keras import TqdmCallback
import tqdm
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import json
import os
import argparse
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
log_mode = enumerate(['QUIET', 'INFO', 'WARNING', 'ERROR', 'DEBUG'])


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


def load_model(model_filename: str, log_mode: log_mode = 0):
    '''
    Loads the model from model.keras.
    '''
    try:
        global_vars.model = keras.models.load_model(model_filename)
        if log_mode >= 1:
            lb_logger.log_info("Model " + model_filename + " loaded.")
        return global_vars.model
    except Exception as e:
        if log_mode >= 3:
            lb_logger.log_error("Failed to load model: " + str(e))


def predict(predict_pd: pd.DataFrame, verbose: int = 1):
    '''
    Predicts the result based on config.
    '''
    prediction = global_vars.model.predict(predict_pd, verbose=verbose)[0]
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


def train(log_mode: log_mode = 0):
    train_data, train_labels = import_train_data("train_data.csv")
    # lb_logger.log_info("Train data: " + str(train_data))
    # lb_logger.log_info("Train labels: " + str(train_labels))

    model = keras.Sequential([
        keras.layers.Dense(6, activation='relu'),
        keras.layers.Dense(3, activation='relu'),
        keras.layers.Dense(6, activation='sigmoid'),
        # keras.layers.Dense(6)
    ])
    epoch_count = 1000

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    verbosity = 0
    if log_mode >= 1:
        lb_logger.log_info('Training...')
        verbosity = 1
    model.fit(train_data, train_labels, epochs=epoch_count,
              verbose=0, callbacks=[TqdmCallback(verbose=verbosity)])

    test_loss, test_acc = model.evaluate(train_data,  train_labels, verbose=0)

    if log_mode >= 1:
        lb_logger.log_info('Test accuracy: {}'.format(test_acc))
        lb_logger.log_info('Test loss: {}'.format(test_loss))

    # save model
    max_model_id = 0
    for filename in os.listdir('models'):
        if filename.startswith('model_') and filename.endswith('.keras'):
            try:
                model_id = int(filename[6:-6])
                if model_id > max_model_id:
                    max_model_id = model_id
            except:
                pass
    model_id = max_model_id + 1
    four_digit_model_id = str(model_id).zfill(4)
    model_filename = 'models/model_' + four_digit_model_id + '.keras'
    model.save(model_filename)
    if log_mode >= 1:
        lb_logger.log_info('Model saved as ' + model_filename)


def EXIT(log_mode: log_mode = 3):
    if log_mode >= 3:
        lb_logger.log_warning('Force exiting...\n\n\n\n')
    raise SystemExit


def load_model_callback(model_id: str, log_mode: log_mode = 0):
    try:
        global_vars.model_filename = "models/model_" + model_id + ".keras"
        load_model(global_vars.model_filename, log_mode)
    except Exception as e:
        if log_mode >= 3:
            lb_logger.log_error("Failed to load model: " + str(e))


def predict_callback(log_mode: log_mode = 0):
    '''
    Predicts the result based on config.
    '''
    try:
        if log_mode < 4:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        predict_pd = pd.DataFrame([[global_vars.config["dir_1"],
                                    global_vars.config["dir_2"],
                                    global_vars.config["dir_3"],
                                    global_vars.config["dir_4"],
                                    global_vars.config["dir_5"],
                                    global_vars.config["dir_6"]]],
                                  columns=['dir_1', 'dir_2',
                                           'dir_3', 'dir_4',
                                           'dir_5', 'dir_6'])

        prediction, max_prediction, prediction_id = predict(
            insert_randomity(global_vars.config['randomness'],
                             predict_pd))

        prdt_str = ''
        for i in range(len(prediction)):
            prdt_str += str(i) + ': ' + str(prediction[i])
            if i != len(prediction) - 1:
                prdt_str += '\n'

        if log_mode >= 4:
            lb_logger.log_info("Predicted result for\n" +
                               'MDL: ' + global_vars.model_filename)
            lb_logger.log_info('INPT:\n' + str(predict_pd))
            lb_logger.log_info('PRDT:\n' + prdt_str)

        if log_mode >= 1:
            lb_logger.log("RSLT: " + str(prediction_id) +
                          ' (' + str(max_prediction) + ')')

        return prediction_id
    except Exception as e:
        if log_mode >= 3:
            lb_logger.log_error("Failed to predict: " + str(e))


def predict_callback_mult(count: int, log_mode: log_mode = 0):
    '''
    Predicts the result based on config.
    '''
    try:
        if log_mode < 4:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        predict_pd = pd.DataFrame([[global_vars.config["dir_1"],
                                    global_vars.config["dir_2"],
                                    global_vars.config["dir_3"],
                                    global_vars.config["dir_4"],
                                    global_vars.config["dir_5"],
                                    global_vars.config["dir_6"]]],
                                  columns=['dir_1', 'dir_2',
                                           'dir_3', 'dir_4',
                                           'dir_5', 'dir_6'])

        result_ids = []
        for i in tqdm.tqdm(range(count), unit='predictions', desc='Predicting'):
            result_ids.append(predict(
                insert_randomity(global_vars.config['randomness'],
                                 predict_pd), verbose=0)[2])

        if log_mode >= 4:
            lb_logger.log_info("Predicted result for\n" +
                               'MDL: ' + global_vars.model_filename)
            lb_logger.log_info('INPT:\n' + str(predict_pd))

        if log_mode >= 1:
            lb_logger.log("RSLT: " + str(result_ids))

        return result_ids
    except Exception as e:
        if log_mode >= 3:
            lb_logger.log_error("Failed to predict: " + str(e))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-p", "--intpath",
    #                     action="Add path to Python's internal path.",
    #                     default="")
    # args = parser.parse_args()
    # if args.intpath != "":
    #     sys.path.append(args.intpath)
    # train()

    load_config()
    load_model_callback("0004")
    predict_callback_mult(100, 4)
