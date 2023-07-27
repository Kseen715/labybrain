import colorama as cr


class LB_Logger:
    def __init__(self):
        pass

    def log(self, message: str):
        print('[LABYBRAIN] ' + cr.Fore.GREEN +
              'TRACE: ' + message + cr.Style.RESET_ALL)

    def log_debug(self, message: str):
        print('[LABYBRAIN] ' + cr.Fore.BLUE +
              'DEBUG: ' + message + cr.Style.RESET_ALL)

    def log_info(self, message: str):
        print('[LABYBRAIN] ' + cr.Style.RESET_ALL +
              'INFO: ' + message + cr.Style.RESET_ALL)

    def log_warning(self, message: str):
        print('[LABYBRAIN] ' + cr.Fore.YELLOW +
              'WARNING: ' + message + cr.Style.RESET_ALL)

    def log_error(self, message: str):
        print('[LABYBRAIN] ' + cr.Fore.RED +
              'ERROR: ' + message + cr.Style.RESET_ALL)

    def log_critical(self, message: str):
        print('[LABYBRAIN] ' + cr.Fore.RED + cr.Style.BRIGHT +
              'CRITICAL: ' + message + cr.Style.RESET_ALL)

    def test(self):
        self.log("Test log message.")
        self.log_debug("Test debug message.")
        self.log_info("Test info message.")
        self.log_warning("Test warning message.")
        self.log_error("Test error message.")
        self.log_critical("Test critical message.")


if __name__ == "__main__":
    LB_Logger().test()
