import copy
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
from google.protobuf.json_format import MessageToDict

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from torch.utils.data import DataLoader
from ctypes import c_float

import grpc
from tqdm import tqdm
from ecosys.evaluation.criterion import ECELoss
from protos.ecosys_pb2 import EnergyInfo, GPUStat, Message, QueryInferenceRequest, QueryInferenceResponse, RetCode, SimpleResponse
from protos.ecosys_pb2_grpc import ModelInferenceServicer, ModelInferenceStub, add_ModelInferenceServicer_to_server
from ecosys.sysstat.gpu import get_gpu_stats_agg, list_of_obj_agg
from ecosys.context.srv_ctx import ServiceContext
from ecosys.utils.message import serialize, deserialize

softmax = torch.nn.Softmax(dim=1)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class NetworkAttachedModel(ModelInferenceServicer):

    req_queue = mp.Queue()
    rsp_queue = mp.Queue()
    executor = ProcessPoolExecutor(max_workers=2+os.cpu_count())
    # gpu_powers = mp.Queue()

    temperature = torch.nn.Parameter(torch.ones(1) * 1.0)

    gpu_indices = [0]

    def __call__(self, input):
        if type(input) is dict:
            return self._forward_single(input)
        
        if type(input) is DataLoader:
            return self._forward_multi(input)

    def __init__(self, model, ctx: ServiceContext) -> None:
        super(NetworkAttachedModel, self).__init__()

        # setup variables
        
        # self.loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(self.loop)
        self.ctx = ctx
        self.executor = ThreadPoolExecutor(max_workers=ctx.cfg.srv_workers)
        # self.gpu_powers = deque([], maxlen=5000)

        if ctx.cfg.srv_backend is None:
            self.stub = None
        else:
            channel = grpc.insecure_channel(ctx.cfg.srv_backend)
            self.stub = ModelInferenceStub(channel)

        self.threshold = mp.Value(c_float, ctx.cfg.srv_threshold)

        # self.req_lock = threading.Lock()
        # self.rsp_lock = threading.Lock()
        # self.req_queue = list()
        # self.rsp_queue = list()

        model.to(ctx.cfg.srv_device)
        model.eval()
        model.share_memory()
        self.temperature = self.temperature.to(ctx.cfg.srv_device)
        # self.engine = model
        self.engine = deepspeed.init_inference(model, replace_method='auto')

        self.gpu_baseline = self._measure_gpu_baseline()

        # self.executor.submit(self.loop.run_until_complete, self.run_tasks())
        # self.loop.run_forever()
        # asyncio.run(self.monitor_gpu_power())
        # asyncio.run(self.delegate_inference())
        # self.executor.submit(self.monitor_gpu_power)
        self.executor.submit(self.delegate_inference)
        # self.executor.submit(self.monitor_gpu_power)
        # sleep(1) # sleep to wait for gpu power collection

    def PushConfidence(self, request, context):
        head = request.head
        body = request.body

        rsp = Message()
        rsp.head.CopyFrom(head)
        rsp.body.simple_response.CopyFrom(SimpleResponse())

        if not body.HasField('push_confidence_request'):
            rsp.body.simple_response.retcode = RetCode.ERR_MISMATCH_MESSAGE
            rsp.body.simple_response.error_message = f"wrong message type {body.WhichOneof('payload')}, should be PushConfidenceRequest"
            return rsp

        req = body.push_confidence_request

        self.threshold = req.threshold

        return rsp
    
    # def QueryInferenceStream(self, request_iterator, context):
    #     for request in request_iterator:
            
    #         head = request.head
    #         body = request.body

    #         rsp = Message()
    #         rsp.head.CopyFrom(head)
    #         rsp.body.query_inference_response.CopyFrom(QueryInferenceResponse())

    #         if not body.HasField('query_inference_request'):
    #             rsp.body.query_inference_response.retcode = RetCode.ERR_MISMATCH_MESSAGE
    #             rsp.body.query_inference_response.error_message = f"wrong message type {body.fields[0].number}, should be 100001"
    #             yield rsp

    #         req = body.query_inference_request

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
    #             # queue.append(ModelInferenceRequest().CopyFrom(request))
                
    #             with self.req_lock:
    #                 self.req_queue.append(request)
    #                 print('queued')

    #             with self.rsp_lock:
    #                 for rsp in self.rsp_queue:
    #                     print('delegated')
    #                     rsp.body.query_inference_response.energy_info.append(energy)
    #                     yield rsp
    #                 self.rsp_queue.clear()
    #             # for rsp in self.stub.QueryInference(iter([request])):
    #             #     print('delegated')
    #             #     yield rsp
    #             # if len(queue) > 100:
    #             #     for rsp in self.stub.QueryInference(iter(queue)):
    #             #         print('delegated')
    #             #         yield rsp
    #             #     queue = list()
    #         else:
    #             print('handled')
    #             rsp.body.query_inference_response.logits = seri(logits)
    #             rsp.body.query_inference_response.energy_info = energy
    #             yield rsp
    #             #     break
    #             # except:
    #             #     sleep(0.001 * 2 ** cnt)
    #             #     cnt += 1
    #             #     continue

    #     if self.req_queue:
    #         for rsp in self.stub.QueryInference(iter(self.req_queue)):
    #             print('delegated')
    #             rsp.body.query_inference_response.energy_info.append(energy)
    #             yield rsp

    # def _create_response(self, request, logits):
    #     rsp = Message()
    #     rsp.head.CopyFrom(request.head)
    #     rsp.body.query_inference_response.CopyFrom(QueryInferenceResponse())

    #     output = BytesIO()
    #     np.save(output, logits.cpu().detach().numpy(), allow_pickle=False)

    #     return ModelInferenceResponse(
    #         head=request.head,
    #         retcode=0,
    #         logits=output.getvalue(),
    #         power=np.mean(self.gpu_powers),
    #     )

    def _forward_local(self, input):

        # for key in input:
        #     print(key, input[key].shape)
        with torch.no_grad():
            output = self.engine(**input)
        if isinstance(output, torch.Tensor):
            logits = output
        else:
            logits = output.logits
        # torch.cuda.synchronize()
        return self.temperature_scale(logits)

    def _forward_multi(self, dataloader):
        pass

    def _monitor_gpu_stats(self, lock):
        stats = []
        while lock.locked():
            gpu_stats = get_gpu_stats_agg(self.gpu_indices)
            stats.append(
                GPUStat(
                    power = gpu_stats['power.draw'],
                    timestamp = gpu_stats['query_time'],
                    utilization = gpu_stats['utilization.gpu'],
                    mem_used = gpu_stats['memory.used'],
                    mem_total = gpu_stats['memory.total'],
                )
            )
            sleep(0.001)

        return stats

    def _measure_gpu_baseline(self):
        lock = threading.Lock()
        with lock:
            future = self.executor.submit(self._monitor_gpu_stats, lock)
            sleep(2)
        gpu_stats = future.result()
        gpu_baseline = GPUStat(
            power = list_of_obj_agg(gpu_stats, 'power', np.mean, self.gpu_indices),
            timestamp = -1,
            utilization = list_of_obj_agg(gpu_stats, 'utilization', np.mean, self.gpu_indices),
            mem_used = list_of_obj_agg(gpu_stats, 'mem_used', np.mean, self.gpu_indices),
            mem_total = list_of_obj_agg(gpu_stats, 'mem_total', np.mean, self.gpu_indices),
        )
        return gpu_baseline

    
    def _forward_single(self, input, label=None):
        
        ## TIME CONSUMING TODO
        lock = threading.Lock()
        with lock:
            future = self.executor.submit(self._monitor_gpu_stats, lock)
            start_time = time()
            logits = self._forward_local(input)
            probabilities = softmax(logits)
            end_time = time()
        
        gpu_stats = future.result()

        # powers = np.array(list(self.gpu_powers))
        # print(start_time, end_time, np.max(powers[:, 0]), np.min(powers[:, 0]))
        # powers = powers[(powers[:, 0] >= start_time) & (powers[:, 0] <= end_time), 1] - self.baseline_power
        # powers = powers.tolist()
        # latency = end_time - start_time

        energy = [EnergyInfo(
            gpu_stats=gpu_stats,
            latency=end_time - start_time,
        )]

        max_prob, _ = torch.max(probabilities, 1)
        mask = max_prob < self.threshold.value

        # print(mask.shape, logits.shape, torch.count_nonzero(mask))
        # print('before', input)
        if torch.count_nonzero(mask) > 0:
            for key in input:
                input[key] = input[key][mask]
                # print(input[key].shape)
            # print('after', input)
        # print(max_prob < self.threshold.value)

        # if torch.max(probabilities).item() < self.threshold.value:
            req = Message()
            
            req.body.query_inference_request.CopyFrom(QueryInferenceRequest(
                input_batch=serialize(input),
            ))

            # input_s = serialize(input)
            # for key in input_s:
            #     print('send', key, hashlib.md5(input_s[key]).hexdigest())
            #     print('de', key, deserialize(input_s[key], self.ctx.cfg.srv_device))
            if label is not None:
                req.body.query_inference_request.label = serialize(label)
            
            rsp = self.stub.QueryInference(req)
            logits[mask, :] = deserialize(rsp.body.query_inference_response.logits, self.ctx.cfg.srv_device)
            energy_info = rsp.body.query_inference_response.energy_info
            energy += energy_info

        total_energy = np.sum([np.sum(info.power * e.latency / len(e.gpu_stats)) for e in energy for info in e.gpu_stats])
        print('total_energy', total_energy)
        print('total_latency', np.sum([info.latency for info in energy]))
        return logits, energy

    def delegate_inference(self):
        print('delegate_inference starting')
        queue = []
        while True:
            with self.req_lock:
                queue = copy.deepcopy(self.req_queue)   
            if queue:
                for rsp in self.stub.QueryInferenceStream(iter(queue)):
                    with self.rsp_lock:
                        self.rsp_queue.append(rsp)
            sleep(0.01)

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


    def QueryInference(self, request, context):
        
        head = request.head
        body = request.body

        rsp = Message()
        rsp.head.CopyFrom(head)
        rsp.body.query_inference_response.CopyFrom(QueryInferenceResponse())

        if not body.HasField('query_inference_request'):
            rsp.body.query_inference_response.retcode = RetCode.ERR_MISMATCH_MESSAGE
            rsp.body.query_inference_response.error_message = f"wrong message type {body.WhichOneof('payload')}, should be QueryInferenceRequest"
            return rsp

        req = body.query_inference_request

        # for key in req.input_batch:
        #     print('reveived', key, torch.load(BytesIO(req.input_batch[key]), self.ctx.cfg.srv_device))

        input = deserialize(dict(req.input_batch), self.ctx.cfg.srv_device)
        # for key in req.input_batch:
        #     print('reveived', key, hashlib.md5(req.input_batch[key]).hexdigest())
        logits, energy = self._forward_single(input)

        rsp.body.query_inference_response.logits = serialize(logits)
        rsp.body.query_inference_response.energy_info.extend(energy)

        return rsp

    # async def run_tasks(self):
    #     await asyncio.wait([
    #         self.loop.create_task(self.monitor_gpu_power()), 
    #         self.loop.create_task(self.delegate_inference()),
    #     ])

    def serve(self):
        # self.delegation_engine = ThreadPoolExecutor(max_workers=n_threads)
        server = grpc.server(self.executor)
        add_ModelInferenceServicer_to_server(self, server)
        server.add_insecure_port(self.ctx.cfg.srv_listen)
        server.start()
        server.wait_for_termination()

    # def monitor_gpu_power(self):
    #     print('monitor_gpu_power starting')
    #     while True:
    #         self.gpu_powers.append([time(), measure_gpu_power([0])])
    #         sleep(0.001)

    # Model Calibration

    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        nll_criterion = torch.nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in tqdm(valid_loader, desc="Training"):
                output = self.model(**input)
                if isinstance(output, torch.Tensor):
                    logits = output
                else:
                    logits = output.logits

                logits_list.append(logits)
                labels_list.append(label)

                # num_batch += 1
                # if num_batch > 10: break
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
        labels = torch.flatten(labels)
        # Calculate NLL and ECE before temperature scaling
        # print(logits.shape, labels.shape)
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        self.ctx.logger.info('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=500)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        # for _ in tqdm(range(10)):
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        self.ctx.logger.info('Optimal temperature: %.3f' % self.temperature.item())
        self.ctx.logger.info('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
