import torch
import logging
import numpy as np
from outliers import smirnov_grubbs as grubbs
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

# from src.models.temperature_scaling import ModelWithTemperature
from src.models.g_layers import ModelWithCalibration
from src.utils.data_processor import processors, output_modes
from src.utils.data_structure import HuggingFaceDataset

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

feature_size = 768
sequence_length = 128
task_name = 'SST-2'
batch_size = 32

filename = __file__
filename = filename.split(".")[0]
fh = logging.FileHandler(f'{filename}_{task_name}.log', mode='w')
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

model_paths = [
    # f"{base_dir}/HuggingFace/distilbert-base-uncased-{task_name}",
    f"/home/oai/efficient-nlp/outputs/distilbert-base-uncased/FineTune_bsz_lr_epoch2_SST-2/checkpoint-1000",
    f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}",
    # "/home/oai/efficient-nlp/outputs/bert-large-uncased/CoLAFineTune_bsz16_lr0.00003_epoch20_CoLA/finetuned_quantized_checkpoints/epoch4_step2675",
]

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
    models[key] = ModelWithCalibration(models[key], 2)
    models[key].set_logger(logger)
    models[key].calibrate(train_dataloader)

# -------------  Evaluation WITH Temperature --------------

correct_cnt = dict(zip(model_keys, [0]*len(model_keys)))
coop_cnt = dict(zip(model_keys, [0]*len(model_keys)))
process_cnt = dict(zip(model_keys, [0]*len(model_keys)))

num_labels = 0
th_stats = []
threshold = None
for input, label in tqdm(test_dataloader, desc="Evaluating"):
    # input_ids = data_batch['input_ids'].to(device)
    # attention_mask = data_batch['attention_mask'].to(device)
    # labels = data_batch['labels'].to(device)

    # if num_labels < 20 :
    #     with torch.no_grad():
    #         logits = models["S"](input_ids=input_ids, attention_mask=attention_mask)
    #         probabilities = m(logits).cpu()
    #         th_stats.append(np.max(probabilities.cpu().detach().numpy(), axis=1).tolist())
    #     num_labels += batch_size
    #     continue
    # elif threshold is None:
    #     threshold = np.mean(th_stats)

    # print(threshold)
    # processed = (selected_prob >= threshold) if key in ['S'] else np.array([True]*len(labels.cpu()))
    
    b_size = len(label.cpu())

    mask = np.array([False]*b_size)

    with torch.no_grad():
        for key in model_keys:
            logits = models[key](input)
            # print(logits)
            # logits = output.logits
            probabilities = m(logits).cpu()

            if key in ['S']:
                th_stats += np.max(probabilities.cpu().detach().numpy(), axis=1).tolist()
                data = np.array(th_stats[-(10*batch_size):]).reshape((-1,1))

                # best_bic = np.inf
                # best_gm = None
                # for k in range(2,4):
                #     gm = GaussianMixture(n_components=k, random_state=0).fit(data)
                #     bic = gm.bic(data)
                #     # idx = np.argmax(gm.means_.flatten())
                #     # mu = gm.means_.flatten()[idx]
                #     # var = gm.covariances_.flatten()[idx]

                #     if bic < best_bic:
                #         best_gm = gm
                #         best_bic = bic
                best_gm = GaussianMixture(n_components=2, random_state=0).fit(data)

                # print("cov", gm.covariances_)
                # print("mean", gm.means_)

                idx = np.argmax(best_gm.means_.flatten())
                mu = best_gm.means_.flatten()[idx]
                var = best_gm.covariances_.flatten()[idx]
                logger.info("Model BIC %s, mu %s, var %s", best_gm.bic(data), mu, var)

                threshold = mu - np.sqrt(var)

            model_ans = np.argmax(probabilities.cpu().detach().numpy(), axis=1)
            true_ans = label.cpu().detach().numpy().flatten()

            selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])
        
            # print(probabilities)
            # print(model_ans)

            processed = (selected_prob >= threshold) if key in ['S'] else np.array([True]*b_size)
            
            # outliers_idx =  grubbs.min_test_indices(th_stats[-(10*batch_size):], alpha=0.3)
            # # logger.debug("outliers_idx %s", outliers_idx)
            # outliers_mask = np.array([False]*min(len(th_stats), 10*batch_size))
            # outliers_mask[outliers_idx] = True
            # outliers_mask = outliers_mask[-b_size:]
            # processed = np.array([True]*b_size)
            # if key in ['S']:
            #     processed[outliers_mask] = False

            correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
            coop_cnt[key] += np.count_nonzero((model_ans == true_ans) & (~mask) & processed)
            process_cnt[key] += np.count_nonzero((~mask) & processed)
            mask |= processed

    num_labels += b_size
        
# num_labels -= len(np.array(th_stats).flatten())
logger.info("Mean Probability = %s", np.mean(th_stats))
sns.distplot(th_stats, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.savefig(f'figures/th_stats_{task_name}.png', bbox_inches="tight")

logger.info("  Num examples = %s", num_labels)
# logger.info("  Threshold = %s", threshold)
for key in model_keys:
    logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_keys:
    logger.info("%s correct count %s, percent %d", key, correct_cnt[key], int(correct_cnt[key]/num_labels * 100))
logger.info("***** Collaborative Eval results *****")
for key in model_keys:
    logger.info("%s process count %s, correct count %s, percent %d", key, process_cnt[key], coop_cnt[key], int(coop_cnt[key]/process_cnt[key] * 100) if process_cnt[key] != 0 else 0)

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


