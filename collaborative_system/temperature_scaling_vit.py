import torch
import logging
import numpy as np
from outliers import smirnov_grubbs as grubbs
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torchvision.models as cv_models

import os
import sys

from torch.nn.modules.activation import Threshold
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from ecosys.models.temperature_scaling import ModelWithTemperature
from ecosys.utils.data_processor import processors, output_modes
from ecosys.utils.data_structure import Dataset, HuggingFaceDataset, TorchVisionDataset, ViTDataset
from ecosys.algo.monte_carlo import monte_carlo_bounds
from ecosys.decorators.eval_decorators import model_eval
from ecosys.utils.eval import compute_metrics

from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification

from torch.utils.data import DataLoader, Subset, SequentialSampler, TensorDataset, dataloader
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

feature_size = 768
sequence_length = 128
task_name = 'resnet'
batch_size = 32

filename = __file__
filename = filename.split(".")[0]
fh = logging.FileHandler(f'{filename}_{task_name}.log', mode='a')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

base_dir = "/home/oai/share"
tokenizer = ViTFeatureExtractor.from_pretrained(f"{base_dir}/HuggingFace/vit-base-patch32-384")

model_keys = [
    "S", 
    "M", 
    "L",
]

energy_discount_factor = [
    0.05,
    0.25, 
    0.5, 
    # 1.0,
]

model_paths = [
    "",
    f"{base_dir}/HuggingFace/vit-base-patch32-384",
    f"{base_dir}/HuggingFace/vit-large-patch32-384",
]

model_energy = dict(zip(model_keys, energy_discount_factor))

model_paths = dict(zip(model_keys, model_paths))

models = dict()
for key in model_keys:
    logger.debug("key %s, path %s", key, model_paths[key])
    models[key] = ViTForImageClassification.from_pretrained(model_paths[key]).to(device) if key != "S" else cv_models.resnet18(pretrained=True)
    models[key].eval()
    models[key].to(device)

# -------------  Dataset Prepare --------------

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

g_cpu = torch.Generator(device='cpu')
g_cpu.manual_seed(2147483647)

index = np.array([x for x in range(len(datasets.ImageNet("/home/oai/share/dataset/.", split="val", transform=preprocess)))])
# index = np.array([x for x in range(1000)])
train_index, test_index = train_test_split(index, test_size=0.6)

# train_sampler = SequentialSampler(train_index)
# test_sampler = SequentialSampler(test_index)

def data_preprocessing():
    val_dataset = TorchVisionDataset(datasets.ImageNet("/home/oai/share/dataset/.", split="val", transform=preprocess))

    # index = np.array([x for x in range(len(val_dataset))])
    # index = np.array([x for x in range(1000)])
    # train_index, test_index = train_test_split(index, test_size=0.6, random_state=0, shuffle=False)
    # train, test = val_dataset[train_index], test_index[test_index]

    train_dataset = Subset(val_dataset, train_index)
    test_dataset = Subset(val_dataset, test_index)

    train_sampler = SequentialSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader

def tans_data_preprocessing():

    val_dataset = ViTDataset(datasets.ImageNet("/home/oai/share/dataset/.", split="val"), tokenizer)

    # index = np.array([x for x in range(len(val_dataset))])
    # index = np.array([x for x in range(1000)])
    # train_index, test_index = train_test_split(index, test_size=0.6, random_state=0, shuffle=False)

    train_dataset = Subset(val_dataset, train_index)
    test_dataset = Subset(val_dataset, test_index)

    train_sampler = SequentialSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader

m = torch.nn.Softmax(dim=1)

train_dataloader, test_dataloader = tans_data_preprocessing()
raw_train_dataloader, raw_test_dataloader = data_preprocessing()

data_limit = 1000

# -------------  Train Temperature --------------

for key in model_keys:
    models[key] = ModelWithTemperature(models[key])
    models[key].set_logger(logger)
    if key != 'S':
        models[key].set_temperature(train_dataloader) 
    else:
        models[key].set_temperature(raw_train_dataloader) 

n_models = len(model_keys)
model_probs = dict(zip(model_keys, [np.array(list()) for _ in range(n_models)]))
num_labels = 0
with torch.no_grad():
    # for input, label, raw_input, _ in tqdm(zip(train_dataloader, raw_train_dataloader), desc="Find Threshold"):
    for d, raw_d in tqdm(zip(train_dataloader, raw_train_dataloader), desc="Find Threshold"):
        input, label = d
        raw_input, raw_label = raw_d
        num_labels += len(label)
        # print(label)
        # print(raw_label)
        assert np.count_nonzero(label.cpu() == raw_label.cpu()) == len(label.cpu())
        for key in model_keys:
            logits = models[key](input) if key != 'S' else models[key](raw_input)
            probabilities = m(logits).cpu().detach().numpy()
            model_ans = np.argmax(probabilities, axis=1).flatten()
            model_probs[key] = np.append(model_probs[key], [p[model_ans[i]] for i, p in enumerate(probabilities)])
# print(num_labels, model_probs)
def total_reward(threshold):
    # threshold = threshold[0]
    # print(threshold)
    reward = 0
    energy = 0
    mask = np.array([False]*num_labels)
    for i, key in enumerate(model_keys):
        processed = (model_probs[key] >= threshold[i]) if key in model_keys[:-1] else np.array([True]*num_labels)
        reward += np.sum(model_probs[key].round(3)[(~mask) & processed])
        energy += model_energy[key]*np.count_nonzero((~mask) & processed)
        mask |= processed

    return (np.around(reward, 0), -energy)
    # return reward - 0.3 * energy

threshold_bounds = monte_carlo_bounds(
        total_reward, 
        [(0.5, 1.0)]*(len(model_keys)-1), 
        [('reward', float), ('energy', float)],
        n=1000,
        tops=20,
        maxiter=20,
    )
mc_threshold = np.mean(
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

with torch.no_grad():


# @model_eval(test_dataloader)
# def eval_monte_carlo(input, label):

    # global num_labels
    # global th_stats
    for d, raw_d in tqdm(zip(test_dataloader, raw_test_dataloader), desc="Evaluating"):
        input, label = d
        raw_input, raw_label = raw_d
        # print(label)
        # print(raw_label)
        assert np.count_nonzero(label.cpu() == raw_label.cpu()) == len(label.cpu())
        num_labels += len(label)
        b_size = len(label.cpu())
        mask = np.array([False]*b_size)

        for i, key in enumerate(model_keys):
            logits = models[key](input) if key != 'S' else models[key](raw_input)
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

# eval_monte_carlo()

# for input, label in tqdm(test_dataloader, desc="Evaluating"):
#     # input_ids = data_batch['input_ids'].to(device)
#     # attention_mask = data_batch['attention_mask'].to(device)
#     # labels = data_batch['labels'].to(device)

#     # if num_labels < 20 :
#     #     with torch.no_grad():
#     #         logits = models["S"](input_ids=input_ids, attention_mask=attention_mask)
#     #         probabilities = m(logits).cpu()
#     #         th_stats.append(np.max(probabilities.cpu().detach().numpy(), axis=1).tolist())
#     #     num_labels += batch_size
#     #     continue
#     # elif threshold is None:
#     #     threshold = np.mean(th_stats)

#     # print(threshold)
#     # processed = (selected_prob >= threshold) if key in ['S'] else np.array([True]*len(labels.cpu()))
    
#     b_size = len(label.cpu())

#     mask = np.array([False]*b_size)

#     with torch.no_grad():
#         for key in model_keys:
#             logits = models[key](input)
#             # print(logits)
#             # logits = output.logits
#             probabilities = m(logits).cpu().detach().numpy()

#             if key in ['S']:
#                 th_stats += np.max(probabilities, axis=1).tolist()
#                 data = np.array(th_stats[-(10*batch_size):]).reshape((-1,1))

#                 # best_bic = np.inf
#                 # best_gm = None
#                 # for k in range(2,4):
#                 #     gm = GaussianMixture(n_components=k, random_state=0).fit(data)
#                 #     bic = gm.bic(data)
#                 #     # idx = np.argmax(gm.means_.flatten())
#                 #     # mu = gm.means_.flatten()[idx]
#                 #     # var = gm.covariances_.flatten()[idx]

#                 #     if bic < best_bic:
#                 #         best_gm = gm
#                 #         best_bic = bic
#                 best_gm = GaussianMixture(n_components=2, random_state=0).fit(data)

#                 # print("cov", gm.covariances_)
#                 # print("mean", gm.means_)

#                 idx = np.argmax(best_gm.means_.flatten())
#                 mu = best_gm.means_.flatten()[idx]
#                 var = best_gm.covariances_.flatten()[idx]
#                 logger.info("Model BIC %s, mu %s, var %s", best_gm.bic(data), mu, var)

#                 threshold = mu - 3*np.sqrt(var)
#                 threshold = mc_threshold

#             model_ans = np.argmax(probabilities, axis=1)
#             true_ans = label.cpu().detach().numpy().flatten()

#             selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])
        
#             # print(probabilities)
#             # print(model_ans)

#             processed = (selected_prob >= threshold) if key in ['S'] else np.array([True]*b_size)
            
#             # outliers_idx =  grubbs.min_test_indices(th_stats[-(10*batch_size):], alpha=0.3)
#             # # logger.debug("outliers_idx %s", outliers_idx)
#             # outliers_mask = np.array([False]*min(len(th_stats), 10*batch_size))
#             # outliers_mask[outliers_idx] = True
#             # outliers_mask = outliers_mask[-b_size:]
#             # processed = np.array([True]*b_size)
#             # if key in ['S']:
#             #     processed[outliers_mask] = False

#             correct_prob[key] += np.sum(selected_prob)
#             process_prob[key] += np.sum(selected_prob[(~mask) & processed])

#             correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
#             coop_cnt[key] += np.count_nonzero((model_ans == true_ans) & (~mask) & processed)
#             process_cnt[key] += np.count_nonzero((~mask) & processed)
#             mask |= processed

#     num_labels += b_size
        
# num_labels -= len(np.array(th_stats).flatten())

for key in model_keys:
    logger.info("%s Mean Probability = %s", key, np.mean(th_stats[key]))
    sns.distplot(th_stats[key], hist=True, kde=True, 
                bins=int(180/5), 
                # color = 'darkblue', 
                label=key,
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4})

# logger.info("%s Mean Probability = %s", key, np.mean(th_stats))
# sns.distplot(th_stats, hist=True, kde=True, 
#             bins=int(180/5), color = 'darkblue', 
#             hist_kws={'edgecolor':'black'},
#             kde_kws={'linewidth': 4})
plt.legend()
plt.savefig(f'figures/th_stats_{task_name}.png', bbox_inches="tight")

logger.info("  Num examples = %s", num_labels)
logger.info("  Threshold = %s", mc_threshold)
for key in model_keys:
    logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_keys:
    logger.info("%s correct count %s, percent %d, prob %s", key, correct_cnt[key], int(correct_cnt[key]/num_labels * 100), correct_prob[key])
logger.info("***** Collaborative Eval results *****")
for key in model_keys:
    logger.info("%s process count %s, correct count %s, percent %d, prob %s", key, process_cnt[key], coop_cnt[key], int(coop_cnt[key]/process_cnt[key] * 100) if process_cnt[key] != 0 else 0, process_prob[key])

