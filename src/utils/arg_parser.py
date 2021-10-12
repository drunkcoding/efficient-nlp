import argparse
import deepspeed

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

    def initialize(self):
        self.add_argument(
            "--num_train_epochs",
            default=3.0,
            type=float,
            help="Total number of training epochs to perform.",
        )
        self.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit float precision instead of 32-bit",
        )
        
    def parse(self):
        self = deepspeed.add_config_arguments(self)
        args = self.parse_args()
        return args

if __name__ == "__main__":
    parser = ArgParser()
    parser.initialize()
    args = parser.parse()
    print(args.num_train_epochs, type(parser))