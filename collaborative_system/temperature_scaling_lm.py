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
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.models.temperature_scaling import ModelWithTemperature
from src.utils.data_processor import processors, output_modes
from src.utils.data_structure import Dataset, HuggingFaceDataset
from src.algo.monte_carlo import monte_carlo_bounds
from src.decorators.model_decorators import model_eval

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, dataloader
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

feature_size = 768
sequence_length = 128
task_name = 'QQP'
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
    # "L",
]

energy_discount_factor = [
    0.25, 
    0.5, 
    # 0.25
]

model_paths = [
    # f"{base_dir}/HuggingFace/distilbert-base-uncased-{task_name}",
    f"/home/oai/efficient-nlp/outputs/distilbert-base-uncased/QQPFineTune_bsz_lr_epoch2_QQP/checkpoint-9500",
    # f"/home/oai/efficient-nlp/outputs/distilbert-base-uncased/FineTune_bsz_lr_epoch2_SST-2/checkpoint-1000",
    # f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}",
    # f"/home/oai/efficient-nlp/outputs/bert-base-uncased/QNLIFineTune_bsz_lr_epoch2_QNLI/checkpoint-3000",
    f"/home/oai/efficient-nlp/outputs/bert-base-uncased/QQPFineTune_bsz_lr_epoch2_QQP/checkpoint-11000",
]

model_energy = dict(zip(model_keys, energy_discount_factor))

model_paths = dict(zip(model_keys, model_paths))

models = dict()
for key in model_keys:
    logger.debug("key %s, path %s", key, model_paths[key])
    models[key] = AutoModelForSequenceClassification.from_pretrained(model_paths[key]).to(device) # if key != "S" else DistilBertForSequenceClassification.from_pretrained(model_paths[key])
    models[key].eval()

# -------------  Dataset Prepare --------------

processor = processors[task_name.lower()]()
output_mode = output_modes[task_name.lower()]

def data_preprocessing():
    texts = processor.get_dev_tsv(f'/data/GlueData/{task_name}/').reset_index()

    train, test = train_test_split(texts, test_size=0.5)

    encoded_texts = tokenizer(
        train["sentence"].to_list(), 
        padding = 'max_length', 
        truncation = True, 
        max_length=sequence_length, 
        return_tensors = 'pt'
    )
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
model_probs = dict(zip(model_keys, [np.array(list()) for _ in range(n_models)]))
num_labels = 0
with torch.no_grad():
    for input, label in tqdm(train_dataloader, desc="Find Threshold"):
        num_labels += len(label)
        for key in model_keys:
            logits = models[key](input)
            probabilities = m(logits).cpu().detach().numpy()
            model_ans = np.argmax(probabilities, axis=1).flatten()
            model_probs[key] = np.append(model_probs[key], [p[model_ans[i]] for i, p in enumerate(probabilities)])
# print(num_labels, model_probs)
def total_reward(threshold):
    threshold = threshold[0]
    reward = 0
    energy = 0
    mask = np.array([False]*num_labels)
    for key in model_keys:
        processed = (model_probs[key] >= threshold) if key in model_keys[:-1] else np.array([True]*num_labels)
        reward += np.sum(model_probs[key].round(3)[(~mask) & processed])
        energy += model_energy[key]*np.count_nonzero((~mask) & processed)
        mask |= processed

    return (reward, -energy)
    # return reward - 0.3 * energy

mc_threshold = np.mean(monte_carlo_bounds(total_reward, [(0.5, 1.0)], [('reward', float), ('energy', float)]))
logger.info("Threshold Bounds %s", monte_carlo_bounds(total_reward, [(0.5, 1.0)], [('reward', float), ('energy', float)]))
# exit()

# threshold = np.random.random(n_models) + 1
# logger.info("threshold init %s", threshold)
# th_stats = dict(zip(model_keys, [list() for _ in range(n_models)]))
# true_labels = dict(zip(model_keys, [list() for _ in range(n_models)]))
# model_labels = dict(zip(model_keys, [list() for _ in range(n_models)]))
# num_labels = 0
# with torch.no_grad():
#     for key in model_keys:
#         for input, label in tqdm(train_dataloader, desc="Find Threshold"):
#             logits = models[key](input)
#             probabilities = m(logits).cpu().detach().numpy()
#             label = label.cpu().detach().numpy().flatten().tolist()
#             model_ans = np.argmax(probabilities, axis=1).tolist()

#             num_labels += len(label)

#             model_labels[key] += model_ans
#             true_labels[key] += label
#             th_stats[key] += [p[model_ans[i]] for i, p in enumerate(probabilities)]

#             # print(model_labels, true_labels, th_stats)
#             # exit()
# num_labels = int(num_labels / len(model_keys))

# for key in model_keys:
#     model_labels[key] = np.array(model_labels[key])
#     true_labels[key] = np.array(true_labels[key])
#     th_stats[key] = np.array(th_stats[key])

# def total_accuracy(threshold):
#     mask = np.array([False]*num_labels)
#     energy = 0
#     correct_cnt = 0
#     prob_sum = 0
#     for i, key in enumerate(model_keys):
#         candidates = (th_stats[key] >= threshold[i]) if key in model_keys[:-1] else np.array([True]*num_labels)
#         processed = (~mask) & candidates
        
#         n_processed = np.count_nonzero(processed)
#         energy += model_energy[key] * n_processed
#         correct_cnt += np.count_nonzero(model_labels[key][processed] == true_labels[key][processed])
#         prob_sum += np.sum(th_stats[key][processed])
#         mask |= candidates
    
#     val = - np.log(energy / num_labels) * np.log(correct_cnt / num_labels)
#     print(correct_cnt, threshold, val, energy / prob_sum**2)
#     return energy / prob_sum**2

# cons = (
#     {'type': 'ineq', 'fun': lambda x:  x - 0.5},
# )

# res = minimize(total_accuracy, threshold, 
#     method='SLSQP', tol=1e-6, 
#     options={'maxiter':500, 'eps':0.1},
#     bounds=((0.5,1.0),(0.5,1.0)),
#     constraints=cons,
# )
# threshold = res.x[0]

# logger.info("minimize %s", res)

# gm = GaussianMixture(n_components=2, random_state=0).fit(th_stats)
# idx = np.argmax(gm.means_.flatten())
# mu = gm.means_.flatten()[idx]
# var = gm.covariances_.flatten()[idx]
# threshold = mu - 3*np.sqrt(var)
# logger.info("Threshold BIC %s, mu %s, var %s, value %s", gm.bic(th_stats), mu, var, threshold)

# -------------  Evaluation WITH Temperature --------------

correct_cnt = dict(zip(model_keys, [0]*len(model_keys)))
correct_prob = dict(zip(model_keys, [0]*len(model_keys)))
coop_cnt = dict(zip(model_keys, [0]*len(model_keys)))
process_prob = dict(zip(model_keys, [0]*len(model_keys)))
process_cnt = dict(zip(model_keys, [0]*len(model_keys)))

num_labels = 0
th_stats = []
threshold = None

@model_eval(test_dataloader)
def eval_monte_carlo(input, label):

    global num_labels
    global th_stats

    b_size = len(label.cpu())
    mask = np.array([False]*b_size)

    for key in model_keys:
        logits = models[key](input)
        probabilities = m(logits).cpu().detach().numpy()

        if key in ['S']:
            th_stats += np.max(probabilities, axis=1).tolist()

        model_ans = np.argmax(probabilities, axis=1)
        true_ans = label.cpu().detach().numpy().flatten()

        selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])
        processed = (selected_prob >= mc_threshold) if key in ['S'] else np.array([True]*b_size)
        
        correct_prob[key] += np.sum(selected_prob)
        process_prob[key] += np.sum(selected_prob[(~mask) & processed])

        correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
        coop_cnt[key] += np.count_nonzero((model_ans == true_ans) & (~mask) & processed)
        process_cnt[key] += np.count_nonzero((~mask) & processed)
        mask |= processed
    
    num_labels += b_size

eval_monte_carlo()

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

# for key in model_keys:
#     logger.info("%s Mean Probability = %s", key, np.mean(th_stats[key]))
#     sns.distplot(th_stats[key], hist=True, kde=True, 
#                 bins=int(180/5), color = 'darkblue', 
#                 hist_kws={'edgecolor':'black'},
#                 kde_kws={'linewidth': 4})

logger.info("%s Mean Probability = %s", key, np.mean(th_stats))
sns.distplot(th_stats, hist=True, kde=True, 
            bins=int(180/5), color = 'darkblue', 
            hist_kws={'edgecolor':'black'},
            kde_kws={'linewidth': 4})
plt.savefig(f'figures/th_stats_{task_name}.png', bbox_inches="tight")

logger.info("  Num examples = %s", num_labels)
logger.info("  Threshold = %s", threshold)
for key in model_keys:
    logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_keys:
    logger.info("%s correct count %s, percent %d, prob %s", key, correct_cnt[key], int(correct_cnt[key]/num_labels * 100), correct_prob[key])
logger.info("***** Collaborative Eval results *****")
for key in model_keys:
    logger.info("%s process count %s, correct count %s, percent %d, prob %s", key, process_cnt[key], coop_cnt[key], int(coop_cnt[key]/process_cnt[key] * 100) if process_cnt[key] != 0 else 0, process_prob[key])

# # -------------  Evaluation WITHOUT Temperature --------------

# model_labels = []
# true_labels = []

# for data_batch in tqdm(dev_dataloader, desc="Evaluating"):
#     input_ids = data_batch['input_ids'].to(device)
#     # token_type_ids = data_batch['token_type_ids'].to(device)
#     attention_mask = data_batch['attention_mask'].to(device)
#     labels = data_batch['labels'].to(device)

#     true_labels += data_batch['labels'].numpy().flatten().tolist()

#     with torch.no_grad():
#         features = {}
#         preds = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = preds.logits
#         # logger.debug("softmax output %s", m(logits).cpu())
#         model_ans = np.argmax(m(logits).cpu(),axis=1)
#         model_labels += model_ans.flatten().tolist()
        
# true_labels = np.array(true_labels)
# model_labels = np.array(model_labels)

# logger.info("**** Label Stats (Train) ****")
# logger.info("num_labels %s, pred ones %s", len(true_labels), np.count_nonzero(model_labels)) 

# logger.info("**** Model Accuracy Before Temerature ****")
# logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
#     np.corrcoef(true_labels, model_labels)[0,1],
#     len(true_labels),
#     np.count_nonzero(model_labels == true_labels),
#     np.count_nonzero(model_labels == true_labels) / len(true_labels)
# )


