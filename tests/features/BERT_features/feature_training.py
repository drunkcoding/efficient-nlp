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
task_name = 'SST-2'
batch_size = 32


base_dir = "/home/oai/share"
model_name = f"distilbert-base-uncased-{task_name}"
tokenizer = AutoTokenizer.from_pretrained(f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}")
model = AutoModelForSequenceClassification.from_pretrained(f"{base_dir}/HuggingFace/{model_name}")
model.to(device)

fh = logging.FileHandler(f'{filename}_{model_name}.log', mode='w')
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


# @profile
# def main():
def get_features(name):
    def hook(model, input, output):
        if name not in features:
            features[name] = torch.flatten(output.cpu().detach(), start_dim=1)
        else:
            features[name] = torch.vstack((features[name], torch.flatten(output.cpu().detach(), start_dim=1)))
        # features[name] = output.detach()
    return hook

# logger.info("model detail for hook %s", model)
num_layers = 6
for i in range(num_layers):
    model.distilbert.transformer.layer[i].output_layer_norm.register_forward_hook(get_features(i))
    # model.bert.encoder.layer[i].output.LayerNorm.register_forward_hook(get_features(i))
# model.bert.encoder.layer[11].output.LayerNorm.register_forward_hook(get_features(11))
# model.distilbert.transformer.layer[5].output_layer_norm.register_forward_hook(get_features(5))

model.eval()
m = torch.nn.Softmax(dim=1)

train_dataloader = data_preprocessing(train=True)
dev_dataloader = data_preprocessing(train=False)

data_limit = 1000

# -------------  Train KMeans Classifier --------------

features = {}
model_labels = []
true_labels = []

for index, data_batch in enumerate(tqdm(
    train_dataloader, desc="Training"
)):

    if index+1 > data_limit: break
    input_ids = data_batch['input_ids'].to(device)
    # token_type_ids = data_batch['token_type_ids'].to(device)
    attention_mask = data_batch['attention_mask'].to(device)
    labels = data_batch['labels'].to(device)

    true_labels += data_batch['labels'].numpy().flatten().tolist()

    with torch.no_grad():
        # features = {}
        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = preds.logits
        model_ans = np.argmax(m(logits).cpu(),axis=1)
        model_labels += model_ans.flatten().tolist()
        
        # feature = torch.flatten(features['feats'], start_dim=1)

        # if feature_dataset is None:
        #     feature_dataset = feature.cpu()
        # else:
        #     feature_dataset = torch.vstack((feature_dataset, feature.cpu()))

for key in features:
    assert len(features[key]) == len(true_labels)
assert len(model_labels) == len(true_labels)

# test_labels = train_texts['label'][:min(data_limit*batch_size, len(train_texts['label']))]

true_labels = np.array(true_labels)
model_labels = np.array(model_labels)

for key in features:
    logger.info("\n")
    logger.info("**** Layer %s ****", key)
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=100).fit(features[key])
    kmeans.labels_ = np.array(kmeans.labels_)
    centroids = kmeans.cluster_centers_

    percentage = np.count_nonzero(kmeans.labels_ == true_labels) / len(true_labels)

    if percentage > 0.5:
        centroids_labels = np.array([0,1])
    else:
        centroids_labels = np.array([1,0])
        kmeans.labels_ = 1 - np.array(kmeans.labels_)

    logger.debug("centroids_labels %s", centroids_labels)

    logger.info("**** Label Stats (Train) ****")
    logger.info("num_labels %s, kmeans ones %s, pred ones %s", len(true_labels), np.count_nonzero(kmeans.labels_), np.count_nonzero(model_labels)) 

    logger.info("**** MiniBatchKMeans (Train) ****")
    logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
        np.corrcoef(true_labels, kmeans.labels_)[0,1],
        len(true_labels),
        np.count_nonzero(kmeans.labels_ == true_labels),
        np.count_nonzero(kmeans.labels_ == true_labels) / len(true_labels)
    )
    logger.info("**** Model Output (Train)  ****")
    logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
        np.corrcoef(model_labels, kmeans.labels_)[0,1],
        len(model_labels),
        np.count_nonzero(kmeans.labels_ == model_labels),
        np.count_nonzero(kmeans.labels_ == model_labels) / len(model_labels)
    )
    logger.info("**** Predicted Model Output (Train)  ****")
    pred_labels = model_labels[kmeans.labels_ == model_labels]
    cmp_labels = true_labels[kmeans.labels_ == model_labels]
    logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
        np.corrcoef(pred_labels, cmp_labels)[0,1],
        len(pred_labels),
        np.count_nonzero(pred_labels == cmp_labels),
        np.count_nonzero(pred_labels == cmp_labels) / len(cmp_labels)
    )
    # logger.info("**** Cluster Correctness ****")
    # logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
    #     np.corrcoef(model_labels, true_labels)[0,1],
    #     len(model_labels),
    #     np.count_nonzero(true_labels == model_labels),
    #     np.count_nonzero(true_labels == model_labels) / len(true_labels)
    # )


for key in features:
    logger.info("\n")
    logger.info("**** Layer %s ****", key)
    model_labels = []
    cluster_labels = []
    true_labels = []
    for data_batch in tqdm(
        dev_dataloader, desc="Evaluating"
    ):
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
            
            feature = torch.flatten(features[key], start_dim=1).cpu().numpy()

            for f in feature:
                distance = np.linalg.norm(centroids - f ,axis=1)
                # print(distance, distance.shape)
                # print(centroids_labels, centroids_labels.shape)
                assert len(distance) == len(centroids_labels)
                ans = centroids_labels[np.argmin(distance)]
                cluster_labels.append(ans)

    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)
    model_labels = np.array(model_labels)
    assert len(cluster_labels) == len(true_labels)
    # logger.debug("cluster_labels %s", cluster_labels)
    # logger.debug("true_labels %s", true_labels)
    # logger.debug("cluster_labels == true_labels %s", cluster_labels == true_labels)

    logger.info("**** Label Stats (Train) ****")
    logger.info("num_labels %s, kmeans ones %s, pred ones %s", len(true_labels), np.count_nonzero(cluster_labels), np.count_nonzero(model_labels)) 

    logger.info("**** MiniBatchKMeans ****")
    logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
        np.corrcoef(true_labels, cluster_labels)[0,1],
        len(true_labels),
        np.count_nonzero(cluster_labels == true_labels),
        np.count_nonzero(cluster_labels == true_labels) / len(true_labels)
    )
    logger.info("**** Model Output ****")
    logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
        np.corrcoef(model_labels, cluster_labels)[0,1],
        len(model_labels),
        np.count_nonzero(cluster_labels == model_labels),
        np.count_nonzero(cluster_labels == model_labels) / len(model_labels)
    )
    logger.info("**** Predicted Model Output ****")
    pred_labels = model_labels[cluster_labels == model_labels]
    cmp_labels = true_labels[cluster_labels == model_labels]
    logger.info("corrcoef %s, num_labels %s, num_correct %s (%s)", 
        np.corrcoef(pred_labels, cmp_labels)[0,1],
        len(pred_labels),
        np.count_nonzero(pred_labels == cmp_labels),
        np.count_nonzero(pred_labels == cmp_labels) / len(cmp_labels)
    )


# main()