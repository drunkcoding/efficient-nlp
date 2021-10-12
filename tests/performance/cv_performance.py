from tqdm import tqdm
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch
import torch.cuda as cutorch
import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.utils.logger import Logger
from src.utils.data_processor import processors, output_modes
from src.utils.data_structure import HuggingFaceDataset

logger = Logger(__file__, "info", "w")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model_keys = [
    'resnet18',
    'resnet50',
    'resnet152',
    'vgg11',
    'vgg16',
    'vgg19_bn',
    'inception',
    'mobilenet',
]

model_instances = [
    models.resnet18(pretrained=True),
    models.resnet50(pretrained=True),
    models.resnet152(pretrained=True),
    models.vgg11(pretrained=True),
    models.vgg16(pretrained=True),
    models.vgg19_bn(pretrained=True),
    models.inception_v3(pretrained=True),
    models.mobilenet_v2(pretrained=True),
]

for model in model_instances:
    model.to(device)
    model.eval()

models = dict(zip(model_keys, model_instances))

# -------------  Dataset Prepare --------------

dataset = datasets.ImageNet("/home/oai/share/dataset/.", split="val", transform=preprocess)
sampler = SequentialSampler(dataset)
logger.info("n_samples %s", len(dataset))

# performance_schedule = schedule(
#     skip_first=10,
#     wait=5,
#     warmup=1,
#     active=3,
#     repeat=2
# )

import subprocess as sp

record = {
    'bs': list(),
    'key': list(),
    'mem': list(),
    'tol_t': list(),
    'avg_t': list(),
}

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    return np.sum(memory_used_values)

for key in model_keys:
    with torch.no_grad():
        for batch_size in [1, 2, 4, 8, 16 ,32, 64, 128]:
            dataloader = DataLoader(
                    dataset, sampler=sampler, batch_size=batch_size
                )
            # with profile(
            #         activities=[ProfilerActivity.CPU], 
            #         # record_shapes=True,
            #         profile_memory=True,
            #         schedule=performance_schedule,
            #     ) as prof:
            #     # with record_function("model_inference"):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = []
            for data in tqdm(dataloader, desc="Measuring"):
                image = data[0].to(device)
                label = data[1].to(device)
                starter.record()
                _  = models[key](image)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
                # print(dir(cutorch.get_device_properties(device)))
            # print(prof.key_averages())
            record['bs'].append(batch_size)
            record['key'].append(key)
            record['mem'].append(get_gpu_memory())
            record['tol_t'].append(np.sum(timings))
            record['avg_t'].append(np.mean(timings))
            
            logger.info(
                "bs %s; key %s; Mem (MiB) %s; total time (ms) %s; avg time (ms) %s", 
                batch_size, 
                key, 
                get_gpu_memory(), 
                np.sum(timings), 
                np.mean(timings)
            )
            # logger.info("bs %s; key %s;\n\n %s \n\n ", batch_size, key, prof.key_averages().table(sort_by="cuda_time_total"))

df = pd.Dataframe(record)
df.to_csv(os.path.join(os.path.dirname(__file__), f"cv_performance.csv"))