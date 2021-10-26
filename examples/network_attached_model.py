from matplotlib.pyplot import ylabel
import torch
import deepspeed
import numpy as np
from io import BytesIO

from concurrent.futures import ThreadPoolExecutor

import grpc
from ecosys.protos.ecosys_pb2 import MessageType, ModelInferenceResponse, Head
from ecosys.protos.ecosys_pb2_grpc import ModelInferenceServicer, add_ModelInferenceServicer_to_server
from ecosys.sysstat.gpu import measure_gpu_power

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NetworkAttachedModel(torch.nn.Module, ModelInferenceServicer):
    def __init__(self, model):
        super(NetworkAttachedModel, self).__init__()
        self.engine = deepspeed.init_inference(model, replace_method='auto')
        # self.model = model
    
    # def QueryInference(self, request_iterator, context):
    #     for request in request_iterator:
    #         input = {}
    #         for key in request.input_batch:
    #             input[key] = torch.Tensor(np.load(BytesIO(request.input_batch[key]), allow_pickle=False)).int().to(device)
    #         logits = self.engine(**input).logits
    #         logits = logits.cpu().detach().numpy()

    #         output = BytesIO()
    #         np.save(output, logits, allow_pickle=False)

    #         yield ModelInferenceResponse(
    #             head=request.head,
    #             retcode=0,
    #             logits=output.getvalue(),
    #             power=measure_gpu_power(),
    #         )

    def QueryInference(self, request, context):
        input = {}
        for key in request.input_batch:
            input[key] = torch.Tensor(np.load(BytesIO(request.input_batch[key]), allow_pickle=False)).int().to(device)
        # print(input)
        logits = self.engine(**input).logits
        # print(logits)
        logits = logits.cpu().detach().numpy()

        output = BytesIO()
        np.save(output, logits, allow_pickle=False)

        return ModelInferenceResponse(
            head=request.head,
            retcode=0,
            logits=output.getvalue(),
            power=measure_gpu_power(),
        )

    def serve(self, address):
        server = grpc.server(ThreadPoolExecutor(max_workers=40))
        add_ModelInferenceServicer_to_server(self, server)
        server.add_insecure_port(address)
        server.start()
        server.wait_for_termination()
