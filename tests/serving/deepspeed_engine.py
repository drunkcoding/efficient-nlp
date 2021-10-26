import deepspeed

import torch
import logging
import numpy as np
from outliers import smirnov_grubbs as grubbs
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize

import os
import sys

from torch.nn.modules.activation import Threshold
# sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import ecosys
from ecosys.decorators.profile import profile_flops
from ecosys.models.temperature_scaling import ModelWithTemperature
from ecosys.utils.data_processor import processors, output_modes
from ecosys.utils.data_structure import Dataset, HuggingFaceDataset
from ecosys.algo.monte_carlo import monte_carlo_bounds
from ecosys.decorators.eval_decorators import model_eval
from ecosys.utils.eval import compute_metrics

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, dataloader
from sklearn.model_selection import train_test_split

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

feature_size = 768
sequence_length = 128
task_name = 'CoLA'
batch_size = 32

filename = __file__
filename = filename.split(".")[0]
fh = logging.FileHandler(f'{filename}_{task_name}.log', mode='a')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = "/home/oai/share"
tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}")


model_keys = [
    "S", 
    "M", 
    "L",
]

energy_discount_factor = [
    1, 
    10, 
    50,
]

model_paths = [
    f"{base_dir}/HuggingFace/distilbert-base-uncased-{task_name}",
    f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}",
    "/home/oai/efficient-nlp/outputs/bert-large-uncased/FineTune_bsz_lr_epoch4_CoLA/checkpoint-132",
]

model_energy = dict(zip(model_keys, energy_discount_factor))

model_paths = dict(zip(model_keys, model_paths))

models = dict()
for key in model_keys:
    logger.debug("key %s, path %s", key, model_paths[key])
    model = AutoModelForSequenceClassification.from_pretrained(model_paths[key]).to(device)
    models[key] = deepspeed.init_inference(model, mp_size=1, replace_method='auto') 
    models[key].eval()

# -------------  Dataset Prepare --------------

processor = processors[task_name.lower()]()
output_mode = output_modes[task_name.lower()]

def data_preprocessing():
    texts = processor.get_dev_tsv(f'/data/GlueData/{task_name}/').reset_index()

    train, test = train_test_split(texts, test_size=0.5, random_state=0)

    encoded_texts = tokenizer(
        train["sentence"].to_list(), 
        padding = 'max_length', 
        truncation = True, 
        max_length=sequence_length, 
        return_tensors = 'pt'
    )
    # print(encoded_texts)
    # exit()
    dataset = HuggingFaceDataset(encoded_texts, torch.tensor(train['label'].to_list()))
    sampler = SequentialSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    encoded_texts = tokenizer(
        test["sentence"].to_list(), 
        padding = 'max_length', 
        truncation = True, 
        max_length=sequence_length, 
        return_tensors = 'pt'
    )
    dataset = HuggingFaceDataset(encoded_texts, torch.tensor(test['label'].to_list()))
    sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    return train_dataloader, test_dataloader

m = torch.nn.Softmax(dim=1)

train_dataloader, test_dataloader = data_preprocessing()

data_limit = 1000

# -------------  Train Temperature --------------

for key in model_keys:
    models[key] = ModelWithTemperature(models[key])
    models[key].set_logger(logger)
    models[key].set_temperature(train_dataloader)

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
            model_probs[key] += [[p[model_ans[i]], int(model_ans[i] == label[i])] for i, p in enumerate(probabilities)]

for key in model_keys:
    model_probs[key] = np.array(model_probs[key])

def total_reward(threshold):
    reward = 0
    energy = 0
    mask = np.array([False]*num_labels)
    for i, key in enumerate(model_keys):
        processed = (model_probs[key][:, 0] >= threshold[i]) if key in model_keys[:-1] else np.array([True]*num_labels)
        reward += np.around(np.sum(model_probs[key][(~mask) & processed, 1]) / 10.0) * 10
        energy += model_energy[key]* np.count_nonzero(~mask) # np.count_nonzero((~mask) & processed)
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
# exit()

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

        selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])
        processed = (selected_prob >= mc_threshold[i]) if key in model_keys[:-1] else np.array([True]*b_size)
        
        correct_prob[key] += np.sum(selected_prob)
        process_prob[key] += np.sum(selected_prob[(~mask) & processed])

        correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
        coop_cnt[key] += np.count_nonzero((model_ans == true_ans) & (~mask) & processed)
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
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4})

plt.legend()
plt.savefig(f'figures/th_stats_{task_name}.png', bbox_inches="tight")

logger.info("  Num examples = %s", num_labels)
logger.info("  Threshold = %s", mc_threshold)
for key in model_keys:
    logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_keys:
    logger.info("%s correct count %s, percent %d, prob %s", key, correct_cnt[key], np.around(correct_cnt[key]/float(num_labels) * 100, 3), correct_prob[key])
logger.info("***** Collaborative Eval results *****")
for key in model_keys:
    logger.info("%s process count %s, correct count %s, percent %d, prob %s", key, process_cnt[key], coop_cnt[key], np.around(coop_cnt[key]/float(process_cnt[key]) * 100, 3) if process_cnt[key] != 0 else 0, process_prob[key])
