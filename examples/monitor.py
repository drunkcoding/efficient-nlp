
import grpc
from ecosys.context.arg_parser import ArgParser
from ecosys.context.srv_ctx import ServiceContext
from ecosys.service.handler import ReportMetricsHandler
from ecosys.sysstat.gpu import get_gpu_stats_agg
from time import sleep
from google.protobuf.json_format import MessageToDict, ParseDict

from protos.ecosys_pb2 import GPUStat
from protos.ecosys_pb2_grpc import CoordinatorStub

cnt = 0
sleep_time = 0.1
queue = []

args = ArgParser().parse()
ctx = ServiceContext(args.cfg)

coord_stub = CoordinatorStub(
            grpc.insecure_channel(ctx.cfg.coord_addr))

while True:
    gpu_stats = get_gpu_stats_agg(0)
    print(gpu_stats)
    queue.append(
        GPUStat(
            power=float(gpu_stats['power.draw']),
            timestamp=float(gpu_stats['query_time']),
            utilization=float(gpu_stats['utilization.gpu']),
            mem_used=float(gpu_stats['memory.used']),
            mem_total=float(gpu_stats['memory.total']),
        )
    )
    print(queue)
    cnt += 1
    print(cnt, cnt % int(1/sleep_time))
    if cnt % int(1/sleep_time) == 0:
        handler = ReportMetricsHandler(ctx)
        handler.make_request()
        handler.req().model_name = ctx.cfg.model_name
        handler.req().gpu_stats.extend(queue)
        # handler.req().gpu_stats.extend([ParseDict(v, GPUStat()) for i, v in enumerate(queue)])
        queue.clear()

        print(handler.req_msg)

        coord_stub.ReportMetrics.future(handler.req_msg)
    # sleep(sleep_time)