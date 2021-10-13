from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch
import torch.cuda as cutorch
import numpy as np
import pandas as pd

import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from ecosys.utils.logger import Logger
from ecosys.utils.data_processor import processors, output_modes
from ecosys.utils.data_structure import HuggingFaceDataset

logger = Logger(__file__, "info", "w")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

feature_size = 768
sequence_length = 128
task_name = 'QQP'
batch_size = 32

base_dir = "/home/oai/share"
tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/HuggingFace/bert-base-uncased")

model_keys = [
    "Distil", 
    "Base", 
    "Large",
]

model_paths = [
    f"{base_dir}/HuggingFace/distilbert-base-uncased",
    f"{base_dir}/HuggingFace/bert-base-uncased",
    f"{base_dir}/HuggingFace/bert-large-uncased",
]

model_paths = dict(zip(model_keys, model_paths))

models = dict()
for key in model_keys:
    logger.debug("key %s, path %s", key, model_paths[key])
    models[key] = AutoModelForSequenceClassification.from_pretrained(model_paths[key]).to(device)
    models[key].eval()

# -------------  Dataset Prepare --------------

processor = processors[task_name.lower()]()
output_mode = output_modes[task_name.lower()]

def fill_mask(sentence):
    words = sentence.split()
    rnd_idx = np.random.randint(0,len(words))
    words[rnd_idx] = "[MASK]"
    return ' '.join(words)

texts = processor.get_train_tsv(f'/data/GlueData/{task_name}/').reset_index()
texts["sentence"] = texts["sentence"].apply(fill_mask)
encoded_texts = tokenizer(
    texts["sentence"].to_list(), 
    padding = 'max_length', 
    truncation = True, 
    max_length=sequence_length, 
    return_tensors = 'pt'
)
dataset = HuggingFaceDataset(encoded_texts, torch.tensor(texts['label'].to_list()))
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
    # return np.sum(memory_used_values)
    return memory_used_values[-1]

for key in model_keys:
    with torch.no_grad():
        for batch_size in [1, 2, 4, 8, 16 ,32, 64, 128, 256, 512]:
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
            for input, _ in tqdm(dataloader, desc="Measuring"):
                starter.record()
                _  = models[key](**input)
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

df = pd.DataFrame(record)
df.to_csv(os.path.join(os.path.dirname(__file__), f"lm_performance_{task_name}.csv"))