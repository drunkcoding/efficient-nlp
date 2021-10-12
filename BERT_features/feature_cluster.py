import logging
import torch
import os
import sys
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = "false"

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from memory_profiler import profile
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
model_name = f"bert-base-uncased-{task_name}"
tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}")
model = AutoModelForSequenceClassification.from_pretrained(f"{base_dir}/HuggingFace/{model_name}")
model.to(device)

fh = logging.FileHandler(f'{filename}_{model_name}.log', mode='w')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


# -------------  Dataset Prepare --------------
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

processor = processors[task_name.lower()]()
output_mode = output_modes[task_name.lower()]

# train_texts = processor.get_train_tsv(f'/data/GlueData/{task_name}/').reset_index()
train_texts = processor.get_dev_tsv(f'/data/GlueData/{task_name}/').reset_index()
if len(train_texts.index) > batch_size*1000:
    train_texts = train_texts.sample(n=batch_size*1000).reset_index()
encoded_train_texts = tokenizer(
    train_texts["sentence"].to_list(), 
    padding = 'max_length', 
    truncation = True, 
    max_length=sequence_length, 
    return_tensors = 'pt'
)
train_dataset = Dataset(encoded_train_texts, torch.tensor(train_texts['label']))
train_sampler = SequentialSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=batch_size
)

# @profile
# def main():
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

logger.info("model detail for hook %s", model)
# model.distilbert.transformer.layer[5].output_layer_norm.register_forward_hook(get_features('feats'))
model.bert.encoder.layer[11].output.LayerNorm.register_forward_hook(get_features('feats'))

model.eval()
m = torch.nn.Softmax(dim=1)

feature_dataset = None
model_labels = []
true_labels = []

data_limit = 500

for index, data_batch in enumerate(tqdm(
    train_dataloader, desc="Evaluating"
)):

    if index+1 > data_limit: break
    input_ids = data_batch['input_ids'].to(device)
    # token_type_ids = data_batch['token_type_ids'].to(device)
    attention_mask = data_batch['attention_mask'].to(device)
    labels = data_batch['labels'].to(device)

    true_labels += data_batch['labels'].numpy().flatten().tolist()

    with torch.no_grad():
        features = {}
        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = preds.logits
        model_ans = np.argmax(m(logits).cpu(),axis=1)
        model_labels += model_ans.flatten().tolist()
        
        feature = torch.flatten(features['feats'], start_dim=1)

        if feature_dataset is None:
            feature_dataset = feature.cpu()
        else:
            # feature_dataset = np.append(feature_dataset, feature.cpu(), axis=0)
            feature_dataset = torch.vstack((feature_dataset, feature.cpu()))
        
        # print(features['feats'].shape, feature.shape, feature_dataset.shape)
        
        # for i in range(batch_size):
        #     feature = torch.flatten(features['feats'][i])
        #     # print(feature.shape, feature[-1])
        #     # assert len(feature) == 768*sequence_length
        #     feature_dataset.append(feature.cpu().numpt)

        # print(preds.logits.shape,)
        # print(features['feats'].shape)
    # print(feature_dataset)
    


print(len(feature_dataset))
print(data_limit*batch_size)
print(len(true_labels))
assert len(feature_dataset) == len(true_labels)
assert len(model_labels) == len(true_labels)

# test_labels = train_texts['label'][:min(data_limit*batch_size, len(train_texts['label']))]

kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=100).fit(feature_dataset)
# kmeans.labels_ = 1-kmeans.labels_

logger.info("**** MiniBatchKMeans ****")
logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
    np.corrcoef(true_labels, kmeans.labels_)[0,1],
    len(true_labels),
    np.count_nonzero(kmeans.labels_ == true_labels),
    np.count_nonzero(kmeans.labels_ == true_labels) / len(true_labels)
)
logger.info("**** Model Output ****")
logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
    np.corrcoef(model_labels, kmeans.labels_)[0,1],
    len(model_labels),
    np.count_nonzero(kmeans.labels_ == model_labels),
    np.count_nonzero(kmeans.labels_ == model_labels) / len(model_labels)
)
logger.info("**** Model Correctness ****")
logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
    np.corrcoef(model_labels, true_labels)[0,1],
    len(model_labels),
    np.count_nonzero(true_labels == model_labels),
    np.count_nonzero(true_labels == model_labels) / len(true_labels)
)

# main()