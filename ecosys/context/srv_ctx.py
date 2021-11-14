import threading
import os

from torch.types import Number
from .logger import Logger
from .cfg_parser import CfgParser
from ..utils.message import NumGenerator

class ServiceContext():
    cfg = CfgParser()

    def __init__(self, cfg_file) -> None:
        self.cfg.parse(cfg_file)
        self.logger = Logger(self.cfg.log_file, self.cfg.log_level, self.cfg.log_size, self.cfg.log_bk)
        self.ctx_id = os.getpid() ^ NumGenerator.randno()

    # _instance = None
    # _lock = threading.Lock()

    # def __new__(cls, *args, **kwargs):
    #     if not cls._instance:
    #         with cls._lock:
    #             if not cls._instance:
    #                 cls._instance = super(ServiceContext, cls).__new__(cls)
    #             return cls._instance