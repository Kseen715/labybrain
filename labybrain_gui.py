import dearpygui.dearpygui as dpg
import dearpygui_ext.logger as dpg_logger
import colorama as cr
import pandas as pd

import labybrain
from labybrain import global_vars
from labybrain import lb_logger


class DPG_Logger:
    def __init__(self, logger_window: dpg_logger.mvLogger):
        self.internalLogger = logger_window

    def log(self, message: str):
        self.internalLogger.log(message)

    def log_debug(self, message: str):
        self.internalLogger.log_debug(message)

    def log_info(self, message: str):
        self.internalLogger.log_info(message)

    def log_warning(self, message: str):
        self.internalLogger.log_warning(message)

    def log_error(self, message: str):
        self.internalLogger.log_error(message)

    def log_critical(self, message: str):
        self.internalLogger.log_critical(message)

    def test(self):
        self.log("Test log message.")
        self.log_debug("Test debug message.")
        self.log_info("Test info message.")
        self.log_warning("Test warning message.")
        self.log_error("Test error message.")
        self.log_critical("Test critical message.")


def save_callback():
    '''
    Saves config to config.json.
    '''
    global_vars.config["randomness"] = dpg.get_value("randomness")
    global_vars.config["dir_1"] = dpg.get_value("dir_1")
    global_vars.config["dir_2"] = dpg.get_value("dir_2")
    global_vars.config["dir_3"] = dpg.get_value("dir_3")
    global_vars.config["dir_4"] = dpg.get_value("dir_4")
    global_vars.config["dir_5"] = dpg.get_value("dir_5")
    global_vars.config["dir_6"] = dpg.get_value("dir_6")
    try:
        labybrain.save_config()
        logger.log("Config saved.")
    except Exception as e:
        logger.log_error("Failed to save config.json: " + str(e))


def load_callback():
    '''
    Loads config from config.json.
    '''
    labybrain.load_config()
    dpg.set_value("randomness", global_vars.config["randomness"])
    dpg.set_value("dir_1", global_vars.config["dir_1"])
    dpg.set_value("dir_2", global_vars.config["dir_2"])
    dpg.set_value("dir_3", global_vars.config["dir_3"])
    dpg.set_value("dir_4", global_vars.config["dir_4"])
    dpg.set_value("dir_5", global_vars.config["dir_5"])
    dpg.set_value("dir_6", global_vars.config["dir_6"])
    logger.log("Config loaded.")


def reset_callback():
    '''
    Resets config to default values.
    '''
    dpg.set_value("randomness", 0.5)
    dpg.set_value("dir_1", 0.0)
    dpg.set_value("dir_2", 0.0)
    dpg.set_value("dir_3", 0.0)
    dpg.set_value("dir_4", 0.0)
    dpg.set_value("dir_5", 0.0)
    dpg.set_value("dir_6", 0.0)
    logger.log("Config reset to default values.")


def dump_cfg_log():
    '''
    Dumps config to console.
    '''
    log = 'Dumping config to console...\n'\
        + "randomness: " \
        + str(labybrain.global_vars.config["randomness"]) + '\n'\
        + "dir_1: " \
        + str(labybrain.global_vars.config["dir_1"]) + '\n'\
        + "dir_2: " \
        + str(labybrain.global_vars.config["dir_2"]) + '\n'\
        + "dir_3: " \
        + str(labybrain.global_vars.config["dir_3"]) + '\n'\
        + "dir_4: " \
        + str(labybrain.global_vars.config["dir_4"]) + '\n'\
        + "dir_5: " \
        + str(labybrain.global_vars.config["dir_5"]) + '\n'\
        + "dir_6: " \
        + str(labybrain.global_vars.config["dir_6"])
    logger.log_info(log)
    lb_logger.log_info(log)


def predict_callback():
    '''
    Predicts the result based on config.
    '''
    try:
        predict_pd = pd.DataFrame([[global_vars.config["dir_1"],
                                    global_vars.config["dir_2"],
                                    global_vars.config["dir_3"],
                                    global_vars.config["dir_4"],
                                    global_vars.config["dir_5"],
                                    global_vars.config["dir_6"]]],
                                  columns=['dir_1', 'dir_2',
                                           'dir_3', 'dir_4',
                                           'dir_5', 'dir_6'])
        load_model_id = dpg.get_value("model_id")
        if load_model_id == "":
            # find the latest model
            import os
            files = os.listdir("models")
            files.sort()
            load_model_id = files[-1].split("_")[1].split(".")[0]
        global_vars.model_filename = "models/model_" + load_model_id + ".keras"
        labybrain.load_model(global_vars.model_filename)
        prediction, max_prediction, prediction_id = labybrain.predict(
            labybrain.insert_randomity(global_vars.config['randomness'],
                                       predict_pd))

        prdt_str = ''
        for i in range(len(prediction)):
            prdt_str += str(i) + ': ' + str(prediction[i])
            if i != len(prediction) - 1:
                prdt_str += '\n'

        logger.log_info("Predicted result for\n" +
                        'MDL: ' + global_vars.model_filename)
        logger.log_info('INPT:\n' + str(predict_pd))
        logger.log_info('PRDT:\n' + prdt_str)
        logger.log("RSLT: " + str(prediction_id) +
                   ' (' + str(max_prediction) + ')')

        labybrain.lb_logger.log_info("Predicted result for\n" +
                                     'MDL: ' + global_vars.model_filename)
        labybrain.lb_logger.log_info('INPT:\n' + str(predict_pd))
        labybrain.lb_logger.log_info('PRDT:\n' + prdt_str)
        labybrain.lb_logger.log("RSLT: " + str(prediction_id) +
                                ' (' + str(max_prediction) + ')')
    except Exception as e:
        logger.log_error("Failed to predict: " + str(e))
        labybrain.lb_logger.log_error("Failed to predict: " + str(e))


if __name__ == "__main__":

    dpg.create_context()
    dpg.create_viewport(title='Labybrain', height=720, width=1280,
                        decorated=True, resizable=False)
    dpg.set_viewport_small_icon("source/favicon.ico")
    dpg.set_viewport_large_icon("source/favicon.ico")

    dpg.setup_dearpygui()

    width, height, channels, data = dpg.load_image(
        "source/logo.png")
    with dpg.texture_registry():
        texture_id = dpg.add_static_texture(width, height, data)

    with dpg.window(label="Input (krutilki)", menubar=False, width=500,
                    height=682, no_resize=True, no_title_bar=True, no_move=True,
                    no_collapse=True):

        dpg.add_slider_float(label="randomness", min_value=0.0,
                             max_value=1.0, default_value=0, tag="randomness",
                             width=300)
        dpg.add_slider_float(label="dir_1", min_value=0.0,
                             max_value=1.0, default_value=0, tag="dir_1",
                             width=300)
        dpg.add_slider_float(label="dir_2", min_value=0.0,
                             max_value=1.0, default_value=0, tag="dir_2",
                             width=300)
        dpg.add_slider_float(label="dir_3", min_value=0.0,
                             max_value=1.0, default_value=0, tag="dir_3",
                             width=300)
        dpg.add_slider_float(label="dir_4", min_value=0.0,
                             max_value=1.0, default_value=0, tag="dir_4",
                             width=300)
        dpg.add_slider_float(label="dir_5", min_value=0.0,
                             max_value=1.0, default_value=0, tag="dir_5",
                             width=300)
        dpg.add_slider_float(label="dir_6", min_value=0.0,
                             max_value=1.0, default_value=0, tag="dir_6",
                             width=300)

        dpg.add_spacer(height=5)
        dpg.add_button(label="Save", callback=save_callback,
                       width=300, tag="save_button")

        dpg.add_spacer(height=5)
        dpg.add_button(label="Force load", callback=load_callback, width=300)
        dpg.add_button(label="Reset", callback=reset_callback, width=300)
        dpg.add_button(label="Dump to console",
                       callback=dump_cfg_log, width=300)
        dpg.add_spacer(height=5)
        dpg.add_input_text(label="Model ID", readonly=False, width=300,
                           tag="model_id",
                           hint="Leave empty to load the latest model")
        dpg.add_button(label="Predict",
                       callback=predict_callback, width=300)
        dpg.add_image(texture_id, pos=[150, 450])

    dpg.add_window(label="Output (log)", pos=[500, 0], menubar=False, width=765,
                   height=682,
                   no_resize=True, no_title_bar=True, no_move=True,
                   no_collapse=True, tag="log_window")
    mvLogger = dpg_logger.mvLogger(parent="log_window")
    mvLogger.log_level = 0

    logger = DPG_Logger(mvLogger)

    load_callback()

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
