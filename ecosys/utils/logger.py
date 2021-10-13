import logging
import sys
from logging import handlers

class Logger():
    def __init__(self, file, level, mode, *args):        
        if level.lower() == "debug":
            self.level = logging.DEBUG
        elif level.lower() == "info":
            self.level = logging.INFO
        else:
            raise ValueError('logging level must be INFO or DEBUG')

        self.logger = logging.getLogger()
        self.logger.setLevel(self.level)

        format = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s [%(lineno)d] -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        self.logger.addHandler(ch)

        filename = file.split(".")[0]
        for arg in args:
            filename += f"_{arg}"

        fh = handlers.RotatingFileHandler(f"{filename}.log", maxBytes=(1048576*5), backupCount=10, mode=mode)
        fh.setFormatter(format)
        self.logger.addHandler(fh)

    def info(self, pattern, *args):
        self.logger.info(pattern, *args)

    def debug(self, pattern, *args):
        self.logger.debug(pattern, *args)
 
if __name__ == "__main__":
    logger = Logger("info", "w")
    logger.info("abcdefg")

