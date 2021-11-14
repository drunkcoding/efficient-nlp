from ecosys.context.arg_parser import ArgParser
from ecosys.service.coordinator import Coordinator
from ecosys.context.srv_ctx import ServiceContext

import threading

if __name__ == "__main__":

    args = ArgParser().parse()
    print(args.cfg)
    ctx = ServiceContext(args.cfg)

    server = Coordinator(ctx)
    server.serve()

    event = threading.Event()
    event.wait()
    
