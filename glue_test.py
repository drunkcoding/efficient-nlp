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

from src.models.temperature_scaling import ModelWithTemperature
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
task_name = 'QQP'
batch_size = 128

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

model_paths = [
    # f"{base_dir}/HuggingFace/distilbert-base-uncased-{task_name}",
    f"/home/oai/efficient-nlp/outputs/distilbert-base-uncased/QQPFineTune_bsz_lr_epoch2_QQP/checkpoint-9500",
    # f"/home/oai/efficient-nlp/outputs/distilbert-base-uncased/FineTune_bsz_lr_epoch2_SST-2/checkpoint-1000",
    # f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}",
    # f"/home/oai/efficient-nlp/outputs/bert-base-uncased/QNLIFineTune_bsz_lr_epoch2_QNLI/checkpoint-3000",
    f"/home/oai/efficient-nlp/outputs/bert-base-uncased/QQPFineTune_bsz_lr_epoch2_QQP/checkpoint-11000",
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
    train = processor.get_dev_tsv(f'/data/GlueData/{task_name}/').reset_index()
    test = processor.get_test_tsv(f'/data/GlueData/{task_name}/').reset_index()

    # train, test = train_test_split(texts, test_size=0.5)

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
    dataset = HuggingFaceDataset(encoded_texts, None)
    sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    return train_dataloader, test_dataloader, train, test

m = torch.nn.Softmax(dim=1)

train_dataloader, test_dataloader, _, test_texts = data_preprocessing()

# -------------  Train Temperature --------------

for key in model_keys:
    models[key] = ModelWithTemperature(models[key])
    models[key].set_logger(logger)
    models[key].set_temperature(train_dataloader)

# -------------  Evaluation WITH Temperature --------------

# correct_cnt = dict(zip(model_keys, [0]*len(model_keys)))
# coop_cnt = dict(zip(model_keys, [0]*len(model_keys)))
process_cnt = dict(zip(model_keys, [0]*len(model_keys)))

num_labels = 0
th_stats = []
threshold = None
test_labels = list()
for input, label in tqdm(test_dataloader, desc="Testing"):
    b_size = len(label.cpu())
    mask = np.array([False]*b_size)
    test_output = np.array([0]*b_size)
    with torch.no_grad():
        for key in model_keys:
            logits = models[key](input)
            probabilities = m(logits).cpu().detach().numpy()

            if key in ['S']:
                th_stats += np.max(probabilities, axis=1).tolist()
                data = np.array(th_stats[-(10*batch_size):]).reshape((-1,1))

                best_gm = GaussianMixture(n_components=2, random_state=0).fit(data)

                idx = np.argmax(best_gm.means_.flatten())
                mu = best_gm.means_.flatten()[idx]
                var = best_gm.covariances_.flatten()[idx]
                # logger.info("Model BIC %s, mu %s, var %s", best_gm.bic(data), mu, var)

                threshold = min(0.95, mu - 3*np.sqrt(var))

            model_ans = np.argmax(probabilities, axis=1)
            # true_ans = label.cpu().detach().numpy().flatten()

            selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])

            processed = (selected_prob >= threshold) if key in ['S'] else np.array([True]*b_size)
            

            test_output[(~mask) & processed] = model_ans.flatten()[(~mask) & processed]

            # correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
            # coop_cnt[key] += np.count_nonzero((model_ans == true_ans) & (~mask) & processed)
            process_cnt[key] += np.count_nonzero((~mask) & processed)
            mask |= processed

    if task_name == "RTE" or task_name == "QNLI":
            test_output = np.where(test_output == 0, "entailment", "not_entailment")

    test_labels += test_output.tolist()
    num_labels += b_size
        
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
# logger.info("***** Eval results *****")
# for key in model_keys:
#     logger.info("%s correct count %s, percent %d", key, correct_cnt[key], int(correct_cnt[key]/num_labels * 100))
# logger.info("***** Collaborative Eval results *****")
# for key in model_keys:
#     logger.info("%s process count %s, correct count %s, percent %d", key, process_cnt[key], coop_cnt[key], int(coop_cnt[key]/process_cnt[key] * 100) if process_cnt[key] != 0 else 0)

logger.info("***** Collaborative Test results *****")
for key in model_keys:
    logger.info("%s process count %s", key, process_cnt[key])

import subprocess
import pandas as pd

data_dir = f"/data/GlueData/{task_name}"
data_file = "test.tsv"
if task_name == 'MNLI-m':
    data_dir = data_dir.split('-')[0]
    data_file = "test_matched.tsv"
elif task_name == 'MNLI-mm':
    data_dir = data_dir.split('-')[0]
    data_file = "test_mismatched.tsv"
result = subprocess.run(['wc', '-l', os.path.join(data_dir, data_file)], stdout=subprocess.PIPE)
result = result.stdout
num_lines = int(result.split()[0])
ids = [x for x in range(num_lines)]
labels = ["entailment"]*num_lines if task_name in ['RTE', 'QNLI', 'MNLI-m', 'MNLI-mm'] else ["1"]*num_lines
test_result = pd.DataFrame({"id": ids, "label_tmp": labels})


test_texts['label'] = test_labels
test_result = test_result.join(test_texts, on='id', how='left', rsuffix='_other')
test_result = test_result.fillna("entailment" if task_name in ['RTE', 'QNLI', 'MNLI-m', 'MNLI-mm'] else "1")
test_result = test_result[['id', 'label']]
test_result['label'] = test_result['label'].astype(str).replace('\.0','',regex=True)
test_result.to_csv(f"TestResult/{task_name}.tsv", sep='\t',index=False)
print(test_result.head())
print(test_result.info())