import torch
from torchvision import datasets, transforms
import torchvision.models as models
from sklearn.cluster import KMeans, MiniBatchKMeans

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import logging
import argparse
from tqdm import tqdm
import numpy as np
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

logger.info("**** Parse Argument ****")

eval_batch_size = 1

filename = __file__.split('.')[:-1][0]

fh = logging.FileHandler(f'{filename}.log', mode='w')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

logger.info("**** Prepare Dataset ****")
logger.info("%s", torch.cuda.is_available())
device = "cuda:1" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageNet("/home/oai/share/dataset/.", split="val", transform=preprocess)
val_sampler = SequentialSampler(val_dataset)
val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, sampler=val_sampler)

print(val_dataset)

base_label = 18

cnt = 0
labels = []
features = []
for data in tqdm(val_loader, desc="Loading"):
    # print(data[0].shape)
    # print(torch.flatten(data[0], start_dim=1).shape)
    cnt += 1
    if cnt > 5000 or len(labels) > 500: break
    if data[1][0] < base_label: continue
    if data[1][0] < base_label+2:
        features.append(torch.flatten(data[0], start_dim=1))
        labels.append(data[1].flatten().tolist())
    else:
        continue
    
features = torch.cat(tuple(features))
labels = np.array(labels).flatten()

# features = [data[0].flatten() for data in tqdm(val_loader, desc="Images Loading")]
# labels = [data[1].flatten() for data in tqdm(val_loader, desc="Labels Loading")]

kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=100).fit(features)
kmeans.labels_ = np.array(kmeans.labels_) + base_label
centroids = kmeans.cluster_centers_

print(kmeans.labels_)
print(labels)
print(centroids)

logger.info("*** Clustering Accuracy")
logger.info("label match %s, percentage %s", 
    np.count_nonzero(kmeans.labels_ == labels),
    np.round(np.count_nonzero(kmeans.labels_ == labels) / len(labels)*100))