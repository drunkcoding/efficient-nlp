import argparse
import threading
import deepspeed

class ArgParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)

        self.add_argument(
            "--cfg",
            required=True,
            type=str,
            help="Path to configuration file.",
        )

    def parse(self):
        # self = deepspeed.add_config_arguments(self)
        args = self.parse_args()
        return args

    # _instance = None
    # _lock = threading.Lock()

    # def __new__(cls, *args, **kwargs):
    #     if not cls._instance:
    #         with cls._lock:
    #             if not cls._instance:
    #                 cls._instance = super(ArgParser, cls).__new__(cls)
    #             return cls._instance

if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse()
    print(args.num_train_epochs, type(parser))