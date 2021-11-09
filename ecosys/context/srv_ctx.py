import threading
from .logger import Logger
from .cfg_parser import CfgParser

class ServiceContext():
    _instance = None
    _lock = threading.Lock()

    cfg = CfgParser()

    def __init__(self, cfg_file) -> None:
        self.cfg.parse(cfg_file)
        self.logger = Logger(self.cfg.log_file, self.cfg.log_level, self.cfg.log_size, self.cfg.log_bk)

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ServiceContext, cls).__new__(cls)
                return cls._instance