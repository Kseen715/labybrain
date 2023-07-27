from logger import LB_Logger
import json


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
        lb_logger.log("Config loaded.")
    except Exception as e:
        lb_logger.log_error("Failed to load config.json: " + str(e))


def save_config():
    '''
    Saves config to config.json.
    '''
    try:
        with open("config.json", "w") as f:
            json.dump(global_vars.config, f)
        lb_logger.log("Config saved.")
    except Exception as e:
        lb_logger.log_error("Failed to save config.json: " + str(e))


if __name__ == "__main__":
    pass
