import copy
import functools
from multiprocessing import Manager
from typing import Any, Tuple
import torch
import deepspeed
import numpy as np
from io import BytesIO
from time import sleep, time
import asyncio
import threading
from collections import deque
import torch.multiprocessing as mp
import os
import hashlib
from google.protobuf.json_format import MessageToDict, ParseDict

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.utils.data import DataLoader
from ctypes import c_float, c_int

import grpc
from tqdm import tqdm
from ecosys.algo.calibratiuon import temperature_scale
from ecosys.decorators.eval import no_grad
from ecosys.evaluation.criterion import ECELoss
from ecosys.service.handler import InferenceHandler, RegisterModelHandler, ReportMetaHandler, ReportMetricsHandler
from ecosys.utils.data_structure import queue2list
from ecosys.utils.database import DatabaseHelper
from protos.ecosys_pb2 import EnergyInfo, GPUStat, Message, InferenceRequest, InferenceResponse, RetCode, SimpleResponse
from protos.ecosys_pb2_grpc import CoordinatorStub, ModelServicer, ModelStub, add_ModelServicer_to_server
from ecosys.sysstat.gpu import get_gpu_stats_agg
from ecosys.context.srv_ctx import ServiceContext
from ecosys.utils.message import serialize, deserialize
from ecosys.utils.sys_utils import kill_child_processes

import mysql.connector

softmax = torch.nn.Softmax(dim=1)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# class Background():
#     def __init__(self, target, args):
#         self.process = mp.Process(target=target, args=args)

#     def __enter__(self):
#         self.process.start()

#     def __exit__(self, exception_type, exception_value, traceback):
#         self.process.terminate()


class NetworkAttachedModel(ModelServicer):

    # req_queue = mp.Queue()
    # rsp_queue = mp.Queue()
    executor = ThreadPoolExecutor(max_workers=10)
    gpu_queue = mp.Queue()
    event = mp.Event()
    event_counter = mp.Value(c_int, 0)

    query_counter = mp.Value(c_int, 0)

    msg_counter = mp.Value(c_int, 0)

    # temperature = torch.nn.Parameter(torch.ones(1) * 1.0)

    def __call__(self, batch):
        return self.forward(batch)

    def __init__(self, model, ctx: ServiceContext) -> None:
        super(NetworkAttachedModel, self).__init__()

        # setup variables

        self.db_helper = mysql.connector.connect(
            host=ctx.cfg.db_addr,
            user=ctx.cfg.db_user,
        )
        # DatabaseHelper(
        #     host=ctx.cfg.db_addr,
        #     user=ctx.cfg.db_user,
        # )
        self.cursor = self.db_helper.cursor(dictionary=True)

        # self.loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(self.loop)
        self.ctx = ctx
        self.executor = ThreadPoolExecutor(max_workers=ctx.cfg.srv_workers)
        # self.gpu_powers = deque([], maxlen=5000)

        self.coord_stub = CoordinatorStub(
            grpc.insecure_channel(ctx.cfg.coord_addr))

        if self.ctx.cfg.srv_backend is None:
            self.stub = None
        else:
            self.channel = grpc.insecure_channel(self.ctx.cfg.srv_backend)
            self.stub = ModelStub(self.channel)

        self.ctx.logger.info('stub created on %s', ctx.cfg.srv_backend)

        self.threshold = ctx.cfg.srv_threshold

        # self.req_lock = threading.Lock()
        # self.rsp_lock = threading.Lock()
        # self.req_queue = list()
        # self.rsp_queue = list()

        self.temperature = torch.nn.Parameter(
            torch.ones(1, device=ctx.cfg.srv_device) * 1.0)
        # print(self.temperature.is_leaf)

        self.engine = model.to(ctx.cfg.srv_device)
        # self.engine = deepspeed.init_inference(model, replace_method='auto').to(ctx.cfg.srv_device)
        self.engine.eval()
        self.engine.share_memory()
        self.ctx.logger.info('inference engine created')
        # self.gpu_baseline = self._measure_gpu_baseline()
        # self.ctx.logger.debug('gpu_baseline %s', self.gpu_baseline)

        # self.executor.submit(self.loop.run_until_complete, self.run_tasks())
        # self.loop.run_forever()
        # asyncio.run(self.monitor_gpu_power())
        # asyncio.run(self.delegate_inference())

        # self.executor.submit(self._monitor_gpu_stats)
        # self.executor.submit(self.delegate_inference)
        # self.executor.submit(self.monitor_gpu_power)
        # sleep(1) # sleep to wait for gpu power collection

    # def PushConfidence(self, request, context):
    #     head = request.head
    #     body = request.body

    #     rsp = Message()
    #     rsp.head.CopyFrom(head)
    #     rsp.body.simple_response.CopyFrom(SimpleResponse())

    #     if not body.HasField('push_confidence_request'):
    #         rsp.body.simple_response.retcode = RetCode.ERR_MISMATCH_MESSAGE
    #         rsp.body.simple_response.error_message = f"wrong message type {body.WhichOneof('payload')}, should be PushConfidenceRequest"
    #         return rsp

    #     req = body.push_confidence_request

    #     self.threshold = req.threshold

    #     return rsp

    # def InferenceStream(self, request_iterator, context):
    #     for request in request_iterator:

    #         head = request.head
    #         body = request.body

    #         rsp = Message()
    #         rsp.head.CopyFrom(head)
    #         rsp.body.inference_response.CopyFrom(InferenceResponse())

    #         if not body.HasField('inference_request'):
    #             rsp.body.inference_response.retcode = RetCode.ERR_MISMATCH_MESSAGE
    #             rsp.body.inference_response.error_message = f"wrong message type {body.fields[0].number}, should be 100001"
    #             yield rsp

    #         req = body.inference_request

    #         start_time = time()
    #         input = deserialize(req.input_batch)
    #         logits = self.forward(input)
    #         del input
    #         probabilities = softmax(logits)
    #         end_time = time()
    #         powers = np.array(list(self.gpu_powers))
    #         powers = powers[(powers[:, 0] >= start_time) & (powers[:, 0] <= end_time), 1] - self.baseline_power
    #         powers = powers.tolist()
    #         latency = end_time - start_time
    #         energy = EnergyInfo(
    #             power=powers,
    #             latency=latency,
    #         )

    #         if torch.max(probabilities).item() < self.threshold:
    #             # queue.append(ModelRequest().CopyFrom(request))

    #             with self.req_lock:
    #                 self.req_queue.append(request)
    #                 print('queued')

    #             with self.rsp_lock:
    #                 for rsp in self.rsp_queue:
    #                     print('delegated')
    #                     rsp.body.inference_response.energy_info.append(energy)
    #                     yield rsp
    #                 self.rsp_queue.clear()
    #             # for rsp in self.stub.Inference(iter([request])):
    #             #     print('delegated')
    #             #     yield rsp
    #             # if len(queue) > 100:
    #             #     for rsp in self.stub.Inference(iter(queue)):
    #             #         print('delegated')
    #             #         yield rsp
    #             #     queue = list()
    #         else:
    #             print('handled')
    #             rsp.body.inference_response.logits = seri(logits)
    #             rsp.body.inference_response.energy_info = energy
    #             yield rsp
    #             #     break
    #             # except:
    #             #     sleep(0.001 * 2 ** cnt)
    #             #     cnt += 1
    #             #     continue

    #     if self.req_queue:
    #         for rsp in self.stub.Inference(iter(self.req_queue)):
    #             print('delegated')
    #             rsp.body.inference_response.energy_info.append(energy)
    #             yield rsp

    # def _create_response(self, request, logits):
    #     rsp = Message()
    #     rsp.head.CopyFrom(request.head)
    #     rsp.body.inference_response.CopyFrom(InferenceResponse())

    #     output = BytesIO()
    #     np.save(output, logits.cpu().detach().numpy(), allow_pickle=False)

    #     return ModelResponse(
    #         head=request.head,
    #         retcode=0,
    #         logits=output.getvalue(),
    #         power=np.mean(self.gpu_powers),
    #     )

    @no_grad
    def forward(self, batch):
        if type(batch) is torch.utils.data.DataLoader:
            outputs = []
            labels = []
            for input, label in tqdm(batch, desc="Forwarding"):
                output = self._forward(input)
                outputs.append(output)
                labels.append(label)
            return outputs, labels
        else:
            return self._forward(batch[0]), batch[1]

    def _forward(self, input):

        # with self.event_counter.get_lock():
        #     if self.event_counter.value == 0:
        #         self.event.set()
        #         self.event_counter.value += 1
        with self.query_counter.get_lock():
            self.query_counter.value += 1
        if type(input) is dict:
            output = self.engine(**input)
        elif type(input) is torch.Tensor:
            output = self.engine(input)
        elif type(input) is list:
            output = self.engine(*input)
        else: 
            raise ValueError(f"{type(input)} not supported")

        if type(output) is not torch.Tensor:
            output = output.logits

        # with self.event_counter.get_lock():
        #     self.event_counter.value -= 1
        #     if self.event_counter.value == 0:
        #         self.event.clear()

        probabilities = softmax(output)
        max_prob, _ = torch.max(probabilities, 1)
        mask = max_prob < self.threshold

        # print(mask)

        if torch.count_nonzero(mask) > 0:
            for key in input:
                input[key] = input[key][mask]

            handler = InferenceHandler(self.ctx)
            handler.make_request()

            # print(input)

            input = serialize(input)
            for key in input:
                handler.req().input_batch[key] = input[key]

            # print(handler.req_msg)

            rsp = self.stub.Inference(handler.req_msg)

            # print(rsp)

            handler.rsp_msg.CopyFrom(rsp)
            output[mask, :] = deserialize(
                handler.rsp().logits, self.ctx.cfg.srv_device)

        return temperature_scale(output, self.temperature)

    def _monitor_gpu_stats(self):

        stub = CoordinatorStub(grpc.insecure_channel(self.ctx.cfg.coord_addr))

        handler = ReportMetricsHandler(self.ctx)
        handler.make_request()
        handler.req().model_name = self.ctx.cfg.model_name
        handler.req().batch_size = self.ctx.cfg.model_bsz
        handler.req().ctx_id = self.ctx.ctx_id

        while True:
            self.event.wait()
            gpu_stats = get_gpu_stats_agg(self.ctx.cfg.srv_device.index)

            # with self.query_counter.get_lock():
            handler.req().num_query = self.query_counter.value
            self.query_counter.value = 0
            handler.req().gpu_stats.CopyFrom(
                GPUStat(
                    power=gpu_stats['power.draw'],
                    utilization=gpu_stats['utilization.gpu'],
                    mem_used=gpu_stats['memory.used'],
                    mem_total=gpu_stats['memory.total'],
                )
            )
            stub.ReportMetrics(handler.req_msg)

            if not self.event.is_set() and self.query_counter.value == 0:
                break

    # def _report_gpu_stats(self):
    #     while True:
    #     # while self.event.wait():
    #         queue = queue2list(self.gpu_queue)
    #         handler = ReportMetricsHandler(self.ctx)
    #         handler.make_request()
    #         handler.req().model_name = self.ctx.cfg.model_name
    #         handler.req().gpu_stats.extend([ParseDict(v, GPUStat()) for i, v in enumerate(queue)])

    #         self.ctx.logger.debug(handler.req_msg)

    #         self.coord_stub.ReportMetrics.future(handler.req_msg)
    #         # handler.rsp_msg.CopyFrom(rsp)

    #         # self.ctx.logger.info('%s ret:%s, err:%s', type(handler.req()), handler.rsp().rc.retcode, handler.rsp().rc.error_message)
    #         sleep(1)

    # def _measure_gpu_baseline(self):

    #     lock = mp.Lock()
    #     queue = mp.Queue()

    #     lock.acquire()
    #     pid = os.fork()

    #     if pid == 0:
    #         print('child', pid)
    #         self._monitor_gpu_stats(queue, lock)
    #         os._exit(0)

    #     sleep(2)
    #     lock.release()
    #     print('sleep', pid)
    #     os.waitpid(pid, 0)
    #     print('wait')

    #     queue = queue2list(queue)

    #     print(queue)

    #     gpu_stats = [ParseDict(data, GPUStat()) for data in queue]
    #     gpu_baseline = GPUStat(
    #         power=list_of_obj_agg(gpu_stats, 'power',
    #                                 np.mean, self.gpu_indices),
    #         timestamp=-1,
    #         utilization=list_of_obj_agg(
    #             gpu_stats, 'utilization', np.mean, self.gpu_indices),
    #         mem_used=list_of_obj_agg(
    #             gpu_stats, 'mem_used', np.mean, self.gpu_indices),
    #         mem_total=list_of_obj_agg(
    #             gpu_stats, 'mem_total', np.mean, self.gpu_indices),
    #     )
    #     # print(gpu_baseline)
    #     return gpu_baseline

    #     # with lock:
    #     #     future = self.pool.apply_async(
    #     #         self._monitor_gpu_stats, (queue, lock, ))
    #     #     sleep(2)

    #     # print(mp.current_process())
    #     # if mp.parent_process() is None:
    #     # future.wait()

    # def _forward_single(self, input, label=None):

    #     # TIME CONSUMING TODO

    #     # energy pull in background

    #     # lock = mp.Lock()
    #     # queue = mp.Queue()

    #     # lock.acquire()
    #     # pid = os.fork()

    #     # if pid == 0:
    #     #     print('child', pid)
    #     #     self._monitor_gpu_stats(queue, lock)
    #     #     os._exit(0)

    #     # start_time = time()
    #     logits = self._forward(input)
    #     probabilities = softmax(logits)
    #     # end_time = time()

    #     # lock.release()
    #     # print('model', pid)
    #     # os.waitpid(pid, 0)

    #     # if mp.parent_process() is None:
    #     #     lock = threading.Lock()
    #     #     queue = mp.Queue()
    #     #     w        handler.make_response()ith lock:
    #     #         self.pool.apply_async(self._monitor_gpu_stats, (queue, lock, ))
    #     #         start_time = time()
    #     #         logits = self._forward_local(input)
    #     #         probabilities = softmax(logits)
    #     #         end_time = time()

    #     # queue = queue2list(queue)
    #     # gpu_stats = [ParseDict(data, GPUStat()) for data in queue]

    #     # energy = [EnergyInfo(
    #     #     gpu_stats=gpu_stats,
    #     #     latency=end_time - start_time,
    #     # )]

    #     max_prob, _ = torch.max(probabilities, 1)
    #     mask = max_prob < self.threshold

    #     # print(mask.shape, logits.shape, torch.count_nonzero(mask))
    #     # print('before', input)
    #     if torch.count_nonzero(mask) > 0:
    #         for key in input:
    #             input[key] = input[key][mask]
    #             # print(input[key].shape)
    #         # print('after', input)
    #         # print('delegated', torch.count_nonzero(mask))

    #     # if torch.max(probabilities).item() < self.threshold.value:

    #         handler = InferenceHandler(self.ctx)
    #         handler.make_request()

    #         handler.req().input_batch = serialize(input)

    #         # input_s = serialize(input)
    #         # for key in input_s:
    #         #     print('send', key, hashlib.md5(input_s[key]).hexdigest())
    #         #     print('de', key, deserialize(input_s[key], self.ctx.cfg.srv_device))
    #         if label is not None:
    #             handler.req().label = serialize(label)

    #         rsp = self.stub.Inference(handler.req_msg)
    #         logits[mask, :] = deserialize(
    #             rsp.body.inference_response.logits, self.ctx.cfg.srv_device)
    #         # energy_info = rsp.body.inference_response.energy_info
    #         # energy += energy_info

    #     # total_energy = np.sum([np.sum(info.power * e.latency / len(e.gpu_stats))
    #     #                         for e in energy for info in e.gpu_stats])
    #     # print('total_energy', total_energy)
    #     # print('total_latency', np.sum([info.latency for info in energy]))
    #     return logits, []

    # def delegate_inference(self):
    #     print('delegate_inference starting')
    #     queue = []
    #     while True:
    #         with self.req_lock:
    #             queue = copy.deepcopy(self.req_queue)
    #         if queue:
    #             for rsp in self.stub.InferenceStream(iter(queue)):
    #                 with self.rsp_lock:
    #                     self.rsp_queue.append(rsp)
    #         sleep(0.01)

    # def __delegation_generator(self, request_iterator):
    #     for request in request_iterator:
    #         input = deserialize(request.input_batch)
    #         logits = self.engine(**input).logits
    #         del input

    #         probabilities = softmax(logits)

    #         if torch.max(probabilities).item() < self.threshold:
    #             yield True, None, request
    #         else:
    #             output = BytesIO()
    #             np.save(output, logits.cpu().detach().numpy(), allow_pickle=False)
    #             yield False, output, request

    def Inference(self, request, context):

        handler = InferenceHandler(self.ctx)
        handler.req_msg.CopyFrom(request)
        handler.make_response()

        # print(request)

        if not handler.check_req_type():
            return handler.rsp_msg

        input = deserialize(
            dict(handler.req().input_batch), self.ctx.cfg.srv_device)
        logits = self._forward(input)

        handler.rsp().logits = serialize(logits)
        # handler.rsp().energy_info.extend(energy)

        return handler.rsp_msg

    # async def run_tasks(self):
    #     await asyncio.wait([
    #         self.loop.create_task(self.monitor_gpu_power()),
    #         self.loop.create_task(self.delegate_inference()),
    #     ])

    def monitor(self):
        # self.executor.submit(self._monitor_gpu_stats)
        # self.executor.submit(self._report_gpu_stats)
        mp.Process(target=self._monitor_gpu_stats).start()
        # p2 = mp.Process(target=self._report_gpu_stats)
        # p2.start()

        # pid = os.fork()
        # if pid == 0:
        #     self._monitor_gpu_stats()
        #     self.ctx.logger.fatal("monitor_gpu_stats exited!")

        # pid = os.fork()
        # if pid == 0:
        #     self._report_gpu_stats()
        #     self.ctx.logger.fatal("report_gpu_stats exited!")

    def _register(self):
        handler = RegisterModelHandler(self.ctx)
        handler.make_request()
        handler.req().model_name = self.ctx.cfg.model_name

        rsp = self.coord_stub.RegisterModel(handler.req_msg)
        handler.rsp_msg.CopyFrom(rsp)
        handler.log_rc()

        ret = handler.rsp().rc.retcode

        if ret == RetCode.SUCCESS:
            self.threshold = handler.rsp().threshold
            self.temperature = torch.nn.Parameter(
                torch.ones(1, device=self.ctx.cfg.srv_device) * handler.rsp().temperature)
            self.ctx.logger.info("setup threshold %s, temperature %s", self.threshold, self.temperature)
        return handler.rsp().rc.retcode == RetCode.SUCCESS

    def prepare(self, loader):

        # get confidence
        if self._register():
            return

        # run through validation dataset
        self.event.set()
        outputs_list, labels_list = self.forward(loader)
        self.event.clear()

        outputs = torch.cat(outputs_list).to(self.ctx.cfg.srv_device)
        labels = torch.cat(labels_list).to(self.ctx.cfg.srv_device)
        labels = torch.flatten(labels)

        m = torch.nn.Softmax(dim=1)
        pred = torch.flatten(torch.argmax(m(outputs), dim=1))
        self.ctx.logger.info("%s accuray %s", self.ctx.cfg.model_name, torch.count_nonzero(pred == labels) / len(pred))

        # upload results to coordinator
        handler = ReportMetaHandler(self.ctx)
        handler.make_request()
        handler.req().model_name = self.ctx.cfg.model_name
        handler.req().labels = serialize(labels)
        handler.req().outputs = serialize(outputs)

        rsp = self.coord_stub.ReportMeta(handler.req_msg)
        handler.rsp_msg.CopyFrom(rsp)
        handler.log_rc()

        # wait for computation of confidence
        while not self._register():
            sleep(5)

        # self.threshold = handler.rsp().threshold
        # self.temperature = torch.ones(
        #     1, device=self.ctx.cfg.srv_device) * handler.rsp().temperature

        # torch.nn.Parameter(
        #     torch.ones(1) * handler.rsp().temperature).to(self.ctx.cfg.srv_device)

    def serve(self, blocking=False):
        server = grpc.server(self.executor)
        add_ModelServicer_to_server(self, server)
        server.add_insecure_port(self.ctx.cfg.srv_listen)
        server.start()
        if blocking:
            server.wait_for_termination()

    # def monitor_gpu_power(self):
    #     while True:
    #         self.gpu_powers.append([time(), measure_gpu_power([0])])
    #         sleep(0.001)

    # Model Calibration

    # def set_temperature(self, val_loader):
    #     nll_criterion = torch.nn.CrossEntropyLoss().to(self.ctx.cfg.srv_device)
    #     ece_criterion = ECELoss().to(self.ctx.cfg.srv_device)

    #     # First: collect all the logits and labels for the validation set
    #     self.event.set()
    #     logits_list, labels_list = self.forward(val_loader)
    #     self.event.clear()

    #     logits = torch.cat(logits_list).to(self.ctx.cfg.srv_device)
    #     labels = torch.cat(labels_list).to(self.ctx.cfg.srv_device)
    #     labels = torch.flatten(labels)
    #     # Calculate NLL and ECE before temperature scaling
    #     before_temperature_nll = nll_criterion(logits, labels).item()
    #     before_temperature_ece = ece_criterion(logits, labels).item()
    #     self.ctx.logger.info('%s Before temperature - NLL: %.3f, ECE: %.3f' %
    #                          (self.ctx.cfg.model_name, before_temperature_nll, before_temperature_ece))

    #     # Next: optimize the temperature w.r.t. NLL
    #     # print(self.temperature.is_leaf)
    #     optimizer = torch.optim.LBFGS(
    #         [self.temperature], lr=0.01, max_iter=500)

    #     def eval():
    #         optimizer.zero_grad()
    #         loss = nll_criterion(self.temperature_scale(logits), labels)
    #         loss.backward()
    #         return loss

    #     # for _ in tqdm(range(10)):
    #     optimizer.step(eval)

    #     sql = f"UPDATE {self.ctx.cfg.db_name}.model_info SET temperature={self.temperature.cpu().detach().item()} WHERE model_name='{self.ctx.cfg.model_name}'"
    #     print(sql)
    #     self.cursor.execute(sql)
    #     self.db_helper.commit()

    #     # Calculate NLL and ECE after temperature scaling
    #     after_temperature_nll = nll_criterion(
    #         self.temperature_scale(logits), labels).item()
    #     after_temperature_ece = ece_criterion(
    #         self.temperature_scale(logits), labels).item()
    #     self.ctx.logger.info('%s Optimal temperature: %.3f' %
    #                          (self.ctx.cfg.model_name, self.temperature.item()))
    #     self.ctx.logger.info('%s After temperature - NLL: %.3f, ECE: %.3f' %
    #                          (self.ctx.cfg.model_name, after_temperature_nll, after_temperature_ece))

    # def temperature_scale(self, logits):
    #     """
    #     Perform temperature scaling on logits
    #     """
    #     # Expand temperature to match the size of logits
    #     # temperature = self.temperature.unsqueeze(
    #     #     1).expand(logits.size(0), logits.size(1))
    #     return logits / self.temperature

    # ======= UTILS ==============
