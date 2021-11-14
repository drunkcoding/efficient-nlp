from network_attached_model import NetworkAttachedModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler, dataloader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from ecosys.evaluation.metrics import compute_metrics
from ecosys.decorators.eval import model_eval
from ecosys.algo.monte_carlo import monte_carlo_bounds
from ecosys.utils.data_structure import Dataset, HuggingFaceDataset
from ecosys.utils.data_processor import processors, output_modes
from io import BytesIO
import re
from time import sleep
from typing import Iterable
import grpc
import torch
import logging
import numpy as np
from outliers import smirnov_grubbs as grubbs
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import pandas as pd
from ecosys.context.arg_parser import ArgParser
from ecosys.context.srv_ctx import ServiceContext

os.environ['TOKENIZERS_PARALLELISM'] = "false"


feature_size = 768
sequence_length = 128
task_name = 'SST-2'


args = ArgParser().parse()
ctx = ServiceContext(args.cfg)

batch_size = ctx.cfg.model_bsz

base_dir = "/home/oai/share"
tokenizer = AutoTokenizer.from_pretrained(
    f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}")

# -------------  Dataset Prepare --------------

processor = processors[task_name.lower()]()
output_mode = output_modes[task_name.lower()]


def data_preprocessing():
    # train = processor.get_train_tsv(
    #     f'/data/GlueData/{task_name}/').reset_index()
    texts = processor.get_dev_tsv(f'/data/GlueData/{task_name}/').reset_index()
    train, test = train_test_split(texts, test_size=0.5, random_state=0)

    encoded_texts = tokenizer(
        train["sentence"].to_list(),
        padding='max_length',
        truncation=True,
        max_length=sequence_length,
        return_tensors='pt'
    )
    # print(encoded_texts)
    # exit()
    dataset = HuggingFaceDataset(encoded_texts, torch.tensor(
        train['label'].to_list()), ctx.cfg.srv_device)
    sampler = SequentialSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    encoded_texts = tokenizer(
        test["sentence"].to_list(),
        padding='max_length',
        truncation=True,
        max_length=sequence_length,
        return_tensors='pt'
    )
    dataset = HuggingFaceDataset(encoded_texts, torch.tensor(
        test['label'].to_list()), ctx.cfg.srv_device)
    sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    return train_dataloader, test_dataloader


m = torch.nn.Softmax(dim=1)

train_dataloader, test_dataloader = data_preprocessing()


if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(ctx.cfg.model_path)
    model = NetworkAttachedModel(model, ctx)
    model.monitor()
    model.prepare(train_dataloader)
    # model.serve()
    print('initialized')

    print('data prepared')
    # model.to(ctx.cfg.srv_device)
    outputs_list, labels_list = model(test_dataloader)
    outputs = torch.cat(outputs_list)
    labels = torch.cat(labels_list)
    labels = torch.flatten(labels)
    pred = torch.flatten(torch.argmax(m(outputs), dim=1))
    model.ctx.logger.info("%s accuray %s", model.ctx.cfg.model_name, torch.count_nonzero(pred == labels) / len(pred))
    # print(logits)

# def request_iterator(dataloader, delay) -> Iterable[ModelInferenceRequest]:
#     # print('ssssssssssssssssssss')
#     # requests = []
#     for input, _ in tqdm(dataloader, desc="Inference"):
#         for key in input:
#             output = BytesIO()
#             np.save(output, input[key].cpu().detach().numpy(), allow_pickle=False)
#             input[key] = output.getvalue()
#         req = ModelInferenceRequest(
#             head = Head(),
#             input_batch = input,
#             task_type= TaskType.TASK_CLASSIFICATION,
#         )
#         # print('aaaaaaaaaaaaaaa')
#         yield req
#         # requests.append(req)
#         sleep(delay)
#     # for req in requests:

# channel = grpc.insecure_channel('localhost:50051')
# stub = ModelInferenceStub(channel)

# texts = processor.get_train_tsv(f'/data/GlueData/{task_name}/').reset_index()
# encoded_texts = tokenizer(
#     texts["sentence"].to_list(),
#     padding = 'max_length',
#     truncation = True,
#     max_length=sequence_length,
#     return_tensors = 'pt'
# )
# dataset = HuggingFaceDataset(encoded_texts, torch.tensor(texts['label'].to_list()))
# sampler = SequentialSampler(dataset)


# power_samples = {}
# for batch_size in [1,2,4,8,16,32,64]:
#     dataloader = DataLoader(
#         dataset, sampler=sampler, batch_size=batch_size
#     )
#     power_samples[batch_size] = []
#     cnt = 0
#     # rsp_future = []
#     # for req in request_iterator(dataloader, 0.01):
#     #     rsp = stub.QueryInference.future(req)
#     #     rsp_future.append(rsp)
#     rsp_future = stub.QueryInference(request_iterator(dataloader, 0.01))
#     for response in tqdm(rsp_future, desc='Fetching Result'):
#         rsp = response# .result()
#         power_samples[batch_size].append(rsp.power)
#         cnt += batch_size
#         if cnt % 1000 == 0:
#             logger.info('bsz %s, average power %s', batch_size, np.mean(power_samples[batch_size]))
# for key in power_samples:
#     power_samples[key] = [np.max(power_samples[key])]
# df = pd.DataFrame(power_samples)
# df.to_csv('power_samples_d1e-2.csv')

# power_samples = {}
# batch_size = 1
# dataloader = DataLoader(
#     dataset, sampler=sampler, batch_size=batch_size
# )
# # for delay in [0.001, 0.002]:
# for delay in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
#     power_samples[delay] = []
#     cnt = 0
#     # for rsp in stub.QueryInference(request_iterator(dataloader, delay)):
#     # rsp_future = []
#     # for req in request_iterator(dataloader, delay):
#     #     rsp = stub.QueryInference.future(req)
#     #     rsp_future.append(rsp)
#     rsp_future = stub.QueryInference(request_iterator(dataloader, delay))
#     for response in tqdm(rsp_future, desc='Fetching Result'):
#         rsp = response#.result()
#         power_samples[delay].append(rsp.power)
#         cnt += batch_size
#         if cnt % 1000 == 0:
#             logger.info('delay %s, average power %s', delay, np.mean(power_samples[delay]))
# for key in power_samples:
#     power_samples[key] = [np.max(power_samples[key])]
# df = pd.DataFrame(power_samples)
# df.to_csv('power_samples_bsz1.csv')

        # logits = torch.Tensor(np.load(BytesIO(rsp.logits), allow_pickle=False))
        # power_samples[batch_size][delay]
# for input, _ in tqdm(test_dataloader, desc="Inference"):
#     for key in input:
#         output = BytesIO()
#         np.save(output, input[key].cpu().detach().numpy(), allow_pickle=False)
#         input[key] = output.getvalue()
#     req = ModelInferenceRequest(
#         head = Head(),
#         input_batch = input,
#         task_type= TaskType.TASK_CLASSIFICATION,
#     )
#     rsp = stub.QueryInference(req)
#     print(rsp)
#     logits = torch.Tensor(np.load(BytesIO(rsp.logits), allow_pickle=False))
#     print(logits)

# requests = request_iterator()
# print(next(requests))


exit()
# -------------  Train Temperature --------------

n_models = len(model_keys)
num_labels = 0

model_probs = dict(zip(model_keys, [list() for _ in range(n_models)]))
with torch.no_grad():
    for input, label in tqdm(train_dataloader, desc="Find Threshold"):
        num_labels += len(label)
        label = label.cpu().detach().numpy().flatten()
        for key in model_keys:
            logits = models[key](input)
            probabilities = m(logits).cpu().detach().numpy()
            model_ans = np.argmax(probabilities, axis=1).flatten()
            model_probs[key] += [[p[model_ans[i]],
                                  int(model_ans[i] == label[i])] for i, p in enumerate(probabilities)]

for key in model_keys:
    model_probs[key] = np.array(model_probs[key])


def total_reward(threshold):
    reward = 0
    energy = 0
    mask = np.array([False]*num_labels)
    for i, key in enumerate(model_keys):
        processed = (model_probs[key][:, 0] >= threshold[i]
                     ) if key in model_keys[:-1] else np.array([True]*num_labels)
        reward += np.around(np.sum(model_probs[key]
                            [(~mask) & processed, 1]) / 10.0) * 10
        # np.count_nonzero((~mask) & processed)
        energy += model_energy[key] * np.count_nonzero(~mask)
        mask |= processed
    return (reward, -energy)


threshold_bounds = monte_carlo_bounds(
    total_reward,
    [(0.5, 1.0)] * (n_models-1),
    [('reward', float), ('energy', float)],
    n=10000,
    tops=40,
    maxiter=15,
)
mc_threshold = np.min(
    threshold_bounds, axis=1
)
logger.info("Threshold Bounds %s", threshold_bounds)

# -------------  Evaluation WITH Temperature --------------

correct_cnt = dict(zip(model_keys, [0]*n_models))
correct_prob = dict(zip(model_keys, [0]*n_models))
coop_cnt = dict(zip(model_keys, [0]*n_models))
process_prob = dict(zip(model_keys, [0]*n_models))
process_cnt = dict(zip(model_keys, [0]*n_models))

num_labels = 0
# th_stats = []
# threshold = None

th_stats = dict(zip(model_keys, [list() for _ in range(n_models)]))


@profile_flops
def model_inference(model, input):
    return model(input)


@model_eval(test_dataloader)
def eval_monte_carlo(input, label):

    global num_labels
    # global th_stats

    b_size = len(label.cpu())
    mask = np.array([False]*b_size)

    for i, key in enumerate(model_keys):
        logits = model_inference(model=models[key], input=input)
        probabilities = m(logits).cpu().detach().numpy()

        # if key in ['S']:
        #     th_stats += np.max(probabilities, axis=1).tolist()
        th_stats[key] += np.max(probabilities, axis=1).tolist()

        model_ans = np.argmax(probabilities, axis=1)
        true_ans = label.cpu().detach().numpy().flatten()

        selected_prob = np.array([p[model_ans[i]]
                                 for i, p in enumerate(probabilities)])
        processed = (
            selected_prob >= mc_threshold[i]) if key in model_keys[:-1] else np.array([True]*b_size)

        correct_prob[key] += np.sum(selected_prob)
        process_prob[key] += np.sum(selected_prob[(~mask) & processed])

        correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
        coop_cnt[key] += np.count_nonzero((model_ans ==
                                          true_ans) & (~mask) & processed)
        process_cnt[key] += np.count_nonzero((~mask) & processed)
        mask |= processed

    num_labels += b_size


eval_monte_carlo()


for key in model_keys:
    logger.info("%s Mean Probability = %s", key, np.mean(th_stats[key]))
    sns.distplot(th_stats[key], hist=True, kde=True,
                 bins=int(180/5),
                 # color = 'darkblue',
                 label=key,
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 4})

plt.legend()
plt.savefig(f'figures/th_stats_{task_name}.png', bbox_inches="tight")

logger.info("  Num examples = %s", num_labels)
logger.info("  Threshold = %s", mc_threshold)
for key in model_keys:
    logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_keys:
    logger.info("%s correct count %s, percent %d, prob %s", key, correct_cnt[key], np.around(
        correct_cnt[key]/float(num_labels) * 100, 3), correct_prob[key])
logger.info("***** Collaborative Eval results *****")
for key in model_keys:
    logger.info("%s process count %s, correct count %s, percent %d, prob %s", key, process_cnt[key], coop_cnt[key], np.around(
        coop_cnt[key]/float(process_cnt[key]) * 100, 3) if process_cnt[key] != 0 else 0, process_prob[key])
