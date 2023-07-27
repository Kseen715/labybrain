import dearpygui.dearpygui as dpg
import dearpygui_ext.logger as dpg_logger
import colorama as cr

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
    global_vars.config["direction_1"] = dpg.get_value("direction_1")
    global_vars.config["direction_2"] = dpg.get_value("direction_2")
    global_vars.config["direction_3"] = dpg.get_value("direction_3")
    global_vars.config["direction_4"] = dpg.get_value("direction_4")
    global_vars.config["direction_5"] = dpg.get_value("direction_5")
    global_vars.config["direction_6"] = dpg.get_value("direction_6")
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
    dpg.set_value("direction_1", global_vars.config["direction_1"])
    dpg.set_value("direction_2", global_vars.config["direction_2"])
    dpg.set_value("direction_3", global_vars.config["direction_3"])
    dpg.set_value("direction_4", global_vars.config["direction_4"])
    dpg.set_value("direction_5", global_vars.config["direction_5"])
    dpg.set_value("direction_6", global_vars.config["direction_6"])
    logger.log("Config loaded.")


def reset_callback():
    '''
    Resets config to default values.
    '''
    dpg.set_value("randomness", 0.5)
    dpg.set_value("direction_1", 0.0)
    dpg.set_value("direction_2", 0.0)
    dpg.set_value("direction_3", 0.0)
    dpg.set_value("direction_4", 0.0)
    dpg.set_value("direction_5", 0.0)
    dpg.set_value("direction_6", 0.0)
    logger.log("Config reset to default values.")


def dump_cfg_log():
    '''
    Dumps config to console.
    '''
    log = 'Dumping config to console...\n'\
        + "randomness: " \
        + str(labybrain.global_vars.config["randomness"]) + '\n'\
        + "direction_1: " \
        + str(labybrain.global_vars.config["direction_1"]) + '\n'\
        + "direction_2: " \
        + str(labybrain.global_vars.config["direction_2"]) + '\n'\
        + "direction_3: " \
        + str(labybrain.global_vars.config["direction_3"]) + '\n'\
        + "direction_4: " \
        + str(labybrain.global_vars.config["direction_4"]) + '\n'\
        + "direction_5: " \
        + str(labybrain.global_vars.config["direction_5"]) + '\n'\
        + "direction_6: " \
        + str(labybrain.global_vars.config["direction_6"])
    logger.log_info(log)
    lb_logger.log_info(log)


def pr():
    logger.log_error('Not implemented yet!')


if __name__ == "__main__":

    dpg.create_context()
    dpg.create_viewport(title='Labybrain', height=720, width=1280,
                        decorated=True)
    dpg.setup_dearpygui()

    with dpg.window(label="Input (krutilki)", menubar=False, width=400,
                    height=682):

        dpg.add_slider_float(label="Randomness", min_value=0.0,
                             max_value=1.0, default_value=0, tag="randomness",
                             width=300)
        dpg.add_slider_float(label="Direction 1", min_value=0.0,
                             max_value=1.0, default_value=0, tag="direction_1",
                             width=300)
        dpg.add_slider_float(label="Direction 2", min_value=0.0,
                             max_value=1.0, default_value=0, tag="direction_2",
                             width=300)
        dpg.add_slider_float(label="Direction 3", min_value=0.0,
                             max_value=1.0, default_value=0, tag="direction_3",
                             width=300)
        dpg.add_slider_float(label="Direction 4", min_value=0.0,
                             max_value=1.0, default_value=0, tag="direction_4",
                             width=300)
        dpg.add_slider_float(label="Direction 5", min_value=0.0,
                             max_value=1.0, default_value=0, tag="direction_5",
                             width=300)
        dpg.add_slider_float(label="Direction 6", min_value=0.0,
                             max_value=1.0, default_value=0, tag="direction_6",
                             width=300)

        dpg.add_spacer(height=5)
        dpg.add_button(label="Save", callback=save_callback,
                       width=300, tag="save_button")

        dpg.add_spacer(height=5)
        dpg.add_button(label="Force load", callback=load_callback, width=300)
        dpg.add_button(label="Reset", callback=reset_callback, width=300)
        dpg.add_button(label="Dump to console",
                       callback=dump_cfg_log, width=300)
        dpg.add_button(label="Predict",
                       callback=pr, width=300)

    mvLogger = dpg_logger.mvLogger()
    mvLogger.log_level = 0
    dpg.set_item_pos(mvLogger.window_id, [765, 0])
    dpg.set_item_height(mvLogger.window_id, 682)
    logger = DPG_Logger(mvLogger)

    load_callback()

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
