import torch
import logging
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from temperature_scaling import ModelWithTemperature
from data_processor import processors, bert_base_model_config, output_modes
from utils.data_structure import Dataset

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

feature_size = 768
sequence_length = 128
task_name = 'CoLA'
batch_size = 32
model_name = f"distilbert-base-uncased-{task_name}"

filename = __file__
filename = filename.split(".")[0]
fh = logging.FileHandler(f'{filename}_{model_name}.log', mode='w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



base_dir = "/home/oai/share"
model_name = f"bert-base-uncased-{task_name}"
tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}")
model = AutoModelForSequenceClassification.from_pretrained(f"{base_dir}/HuggingFace/{model_name}")
model.to(device)

# -------------  Dataset Prepare --------------

processor = processors[task_name.lower()]()
output_mode = output_modes[task_name.lower()]

def data_preprocessing(train=False):
    if train:
        texts = processor.get_train_tsv(f'/data/GlueData/{task_name}/').reset_index()
        if len(texts.index) > batch_size*500:
            texts = texts.sample(n=batch_size*500).reset_index()
    else:
        texts = processor.get_dev_tsv(f'/data/GlueData/{task_name}/').reset_index()
    
    encoded_texts = tokenizer(
        texts["sentence"].to_list(), 
        padding = 'max_length', 
        truncation = True, 
        max_length=sequence_length, 
        return_tensors = 'pt'
    )
    dataset = Dataset(encoded_texts, torch.tensor(texts['label']))
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    return dataloader


model.eval()
m = torch.nn.Softmax(dim=1)

train_dataloader = data_preprocessing(train=True)
dev_dataloader = data_preprocessing(train=False)

data_limit = 1000

# -------------  Train Temperature --------------

model_labels = []
true_labels = []

for data_batch in tqdm( dev_dataloader, desc="Evaluating"):
    input_ids = data_batch['input_ids'].to(device)
    # token_type_ids = data_batch['token_type_ids'].to(device)
    attention_mask = data_batch['attention_mask'].to(device)
    labels = data_batch['labels'].to(device)

    true_labels += data_batch['labels'].numpy().flatten().tolist()

    with torch.no_grad():
        features = {}
        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = preds.logits
        # logger.debug("softmax output %s", m(logits).cpu())
        model_ans = np.argmax(m(logits).cpu(),axis=1)
        model_labels += model_ans.flatten().tolist()
        
true_labels = np.array(true_labels)
model_labels = np.array(model_labels)

logger.info("**** Label Stats (Train) ****")
logger.info("num_labels %s, pred ones %s", len(true_labels), np.count_nonzero(model_labels)) 

logger.info("**** Model Accuracy Before Temerature ****")
logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
    np.corrcoef(true_labels, model_labels)[0,1],
    len(true_labels),
    np.count_nonzero(model_labels == true_labels),
    np.count_nonzero(model_labels == true_labels) / len(true_labels)
)

scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(dev_dataloader)

model_labels = []
true_labels = []

for data_batch in tqdm(dev_dataloader, desc="Evaluating"):
    input_ids = data_batch['input_ids'].to(device)
    # token_type_ids = data_batch['token_type_ids'].to(device)
    attention_mask = data_batch['attention_mask'].to(device)
    labels = data_batch['labels'].to(device)

    true_labels += data_batch['labels'].numpy().flatten().tolist()

    with torch.no_grad():
        features = {}
        logits = scaled_model(input_ids=input_ids, attention_mask=attention_mask)
        # logits = preds.logits
        # logger.debug("softmax output %s", m(logits).cpu())
        model_ans = np.argmax(m(logits).cpu(),axis=1)
        model_labels += model_ans.flatten().tolist()
        
true_labels = np.array(true_labels)
model_labels = np.array(model_labels)

logger.info("**** Model Accuracy After Temerature ****")
logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
    np.corrcoef(true_labels, model_labels)[0,1],
    len(true_labels),
    np.count_nonzero(model_labels == true_labels),
    np.count_nonzero(model_labels == true_labels) / len(true_labels)
)

logger.info("final temperature %s", scaled_model.temperature)