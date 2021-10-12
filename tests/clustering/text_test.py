import logging
import torch
import os
import sys
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = "false"

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from data_processor import processors, bert_base_model_config, output_modes
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = __file__
filename = filename.split(".")[0]

feature_size = 768
sequence_length = 128
task_name = 'RTE'
batch_size = 32


base_dir = "/home/oai/share"
tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}")

filename = __file__.split('.')[:-1][0]

fh = logging.FileHandler(f'{filename}.log', mode='w')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels != None:
            item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

dev_dataloader = data_preprocessing(train=False)

data_limit = 1000

# -------------  Train KMeans Classifier --------------

labels = []
features = []

for data_batch in tqdm(dev_dataloader, desc="Loading"):
    input_ids = data_batch['input_ids']
    label = data_batch['labels']

    features.append(torch.flatten(input_ids, start_dim=1))
    labels.append(label.flatten())

features = torch.cat(tuple(features))
labels = torch.cat(tuple(labels)).numpy()

# features = [data[0].flatten() for data in tqdm(val_loader, desc="Images Loading")]
# labels = [data[1].flatten() for data in tqdm(val_loader, desc="Labels Loading")]

kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=100).fit(features)
kmeans.labels_ = np.array(kmeans.labels_)
centroids = kmeans.cluster_centers_

print(kmeans.labels_)
print(labels)
# print(centroids)

logger.info("*** Clustering Accuracy")
logger.info("label match %s, percentage %s", 
    np.count_nonzero(kmeans.labels_ == labels),
    np.round(np.count_nonzero(kmeans.labels_ == labels) / len(labels)*100))