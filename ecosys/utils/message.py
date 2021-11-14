

from io import BytesIO
from multiprocessing import Value
import os
from google.protobuf.pyext._message import ScalarMapContainer

import numpy as np
import torch

from threading import Lock
import uuid
import random


def _serialize_numpy(data):
    output = BytesIO()
    np.save(output, data, allow_pickle=False)
    return output.getvalue()


def _serialize_tensor(data):
    output = BytesIO()
    torch.save(data, output)
    return output.getvalue()


def _deserialize_numpy(data):
    arr = np.load(BytesIO(data), allow_pickle=False)
    if np.count_nonzero(np.isclose(arr, np.round(arr))) == len(arr.flatten()):
        arr = arr.astype(int)
    return arr


def _deserialize_tensor(data, device):
    # arr = torch.load(BytesIO(data))
    # if torch.count_nonzero(torch.isclose(arr, torch.round(arr))) == len(arr.flatten()):
    #     arr = arr.int()
    return torch.load(BytesIO(data), device)


def serialize(data):
    if type(data) is torch.Tensor:
        return _serialize_tensor(data)

    # dict like data structure
    if type(data) is dict or type(data) is ScalarMapContainer:
        d = {}
        for key in data:
            d[key] = serialize(data[key])
        return d

    raise ValueError(f"{type(data)} serilization not implemented.")


def deserialize(data, device):
    if type(data) is bytes:
        return _deserialize_tensor(data, device)

    # dict like data structure
    if type(data) is dict or type(data) is ScalarMapContainer:
        d = {}
        for key in data:
            d[key] = deserialize(data[key], device)
        return d

    raise ValueError(f"{type(data)} deserilization not implemented.")


class NumGenerator():

    ctx_id = os.getpid()
    lock = Lock()
    flow_no = 1024

    @classmethod
    def flowno(self):
        with self.lock:
            self.flow_no += 1
            return self.flow_no

    @staticmethod
    def session():
        return str(uuid.uuid4())

    @staticmethod
    def randno():
        return random.randint(0, 1e9)


if __name__ == "__main__":
    a = {
        'a': torch.Tensor([1, 2, 3, 4]),
        "b": torch.Tensor([[1, 2, 3, 4]]),
    }
    print(deserialize(serialize(a), torch.device('cpu')))
