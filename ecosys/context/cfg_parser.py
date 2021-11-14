import configparser
import threading
import torch

class CfgParser():

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    
    def parse(self, filename):
        self.config.read(filename)

        # ------ service config ------
        self._parse_service()

        # ------ logging config ------
        self._parse_logging()

        # ------ database config ------
        self._parse_database()

        # ------ model config ------
        self._parse_model()

        # ------ coordinator config ------
        self._parse_coordinator()

    def _parse_coordinator(self):
        if 'coordinator' not in self.config:
            print('coordinator not in config file, skipping...')
            return

        coordinator = self.config['coordinator']

        self.coord_addr = coordinator.get('address')

    def _parse_model(self):
        if 'model' not in self.config:
            print('model not in config file, skipping...')
            return

        model = self.config['model']

        self.model_name = model.get('name')
        self.model_path = model.get('path')
        self.val_path = model.get('validation')
        self.model_type = model.get('type').lower()
        self.model_bsz = model.getint('batch_size')

    def _parse_database(self):
        if 'database' not in self.config:
            print('database not in config file, skipping...')
            return

        database = self.config['database']

        self.db_addr = database.get('address')
        self.db_user = database.get('user')
        self.db_name = database.get('db')

    def _parse_logging(self):
        if 'logging' not in self.config:
            print('logging not in config file, skipping...')
            return

        logging = self.config['logging']

        self.log_level = logging.get('level', 'info')
        self.log_file = logging.get('file')
        self.log_size = logging.getint('size', '5000')
        self.log_bk = logging.getint('backup', '10')
        self.log_enable = self.log_size >= 0 and self.log_bk >= 0

    def _parse_service(self):
        if 'service' not in self.config:
            print('service not in config file, skipping...')
            return
        
        service = self.config['service']

        backend = service.get('backend')
        self.srv_backend = backend if backend.lower() != 'none' else None

        listen = service.get('listen')
        self.srv_listen = listen if listen.lower() != 'none' else None
        
        device_id = service.getint('device', 0)
        self.srv_device = torch.device(f"cuda:{device_id}" if device_id is not None else "cpu")
        self.srv_threshold = service.getfloat('threshold', 0.0)
        self.srv_workers = service.getint('workers', 10)

    def __init__(self) -> None:
        pass

    # _instance = None
    # _lock = threading.Lock()

    # def __new__(cls, *args, **kwargs):
    #     if not cls._instance:
    #         with cls._lock:
    #             if not cls._instance:
    #                 cls._instance = super(CfgParser, cls).__new__(cls)
    #             return cls._instance