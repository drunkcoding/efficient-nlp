import configparser
import threading
import torch

class CfgParser():
    _instance = None
    _lock = threading.Lock()

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    
    def parse(self, filename):
        self.config.read(filename)

        # ------ service config ------
        service = self.config['service']

        backend = service.get('backend')
        self.srv_backend = backend if backend.lower() != 'none' else None

        listen = service.get('listen')
        self.srv_listen = listen if listen.lower() != 'none' else None
        
        device_id = service.getint('device', 0)
        self.srv_device = torch.device(f"cuda:{device_id}" if device_id is not None else "cpu")
        self.srv_threshold = service.getfloat('threshold', 0.0)
        self.srv_workers = service.getint('workers', 10)

        # ------ logging config ------
        logging = self.config['logging']

        self.log_level = logging.get('level', 'info')
        self.log_file = logging.get('file')
        self.log_size = logging.getint('size', '5000')
        self.log_bk = logging.getint('backup', '10')
        self.log_enable = self.log_size >= 0 and self.log_bk >= 0

    def __init__(self) -> None:
        pass

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(CfgParser, cls).__new__(cls)
                return cls._instance