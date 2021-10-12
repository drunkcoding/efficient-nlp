import torch
from torchvision import datasets, transforms
import torchvision.models as models
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.models.temperature_scaling import ModelWithTemperature
from src.utils.data_processor import processors, bert_base_model_config, output_modes
from src.utils.data_structure import Dataset

from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split

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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default=None,
    type=str,
    required=True,
    help="The type of model, e.g., resnet, vgg, incep-resnet, incep-vgg, mobi-resnet, mobi-vgg",
)

args = parser.parse_args()

eval_batch_size = 64

filename = __file__
filename = filename.split(".")[0]

fh = logging.FileHandler(f'{filename}_{args.model}.log', mode='w')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

logger.info("**** Prepare Dataset ****")
logger.info("%s", torch.cuda.is_available())
device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def data_preprocessing():
    val_dataset = datasets.ImageNet("/home/oai/share/dataset/.", split="val", transform=preprocess)

    index = np.array([x for x in range(len(val_dataset))])
    train_index, test_index = train_test_split(index, test_size=0.4)
    # train, test = val_dataset[train_index], test_index[test_index]

    train_sampler = SubsetRandomSampler(train_index)
    train_loader = DataLoader(val_dataset, batch_size=eval_batch_size, sampler=train_sampler)
    test_sampler = SubsetRandomSampler(test_index)
    test_loader = DataLoader(val_dataset, batch_size=eval_batch_size, sampler=test_sampler)

    return train_loader, test_loader

m = torch.nn.Softmax(dim=1)

train_dataloader, test_dataloader = data_preprocessing()

# val_dataset = datasets.ImageNet("/home/oai/share/dataset/.", split="val", transform=preprocess)
# val_sampler = SequentialSampler(val_dataset)
# val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, sampler=val_sampler)

# print(val_dataset)

logger.info("**** Load Models ****")

model_keys = [
    'resnet18',
    'resnet50',
    'resnet152',
    'vgg11',
    'vgg16',
    'vgg19_bn',
    'inception',
    'mobilenet',
]

model_instances = [
    models.resnet18(pretrained=True),
    models.resnet50(pretrained=True),
    models.resnet152(pretrained=True),
    models.vgg11(pretrained=True),
    models.vgg16(pretrained=True),
    models.vgg19_bn(pretrained=True),
    models.inception_v3(pretrained=True),
    models.mobilenet_v2(pretrained=True),
]

for model in model_instances:
    model.to(device)
    model.eval()

models = dict(zip(model_keys, model_instances))

# resnet18 = models.resnet18(pretrained=True)
# resnet50 = models.resnet50(pretrained=True)
# resnet152 = models.resnet152(pretrained=True)
# vgg11 = models.vgg11(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# vgg19_bn = models.vgg19_bn(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)

if args.model == "resnet":
    model_test_keys = ['resnet18', 'resnet50', 'resnet152']
elif args.model == "vgg":
    model_test_keys = ['vgg11', 'vgg16', 'vgg19_bn']
elif args.model == "incep-resnet":
    model_test_keys = ['inception', 'resnet50', 'resnet152']
elif args.model == "incep-vgg":
    model_test_keys = ['inception', 'vgg19_bn']
elif args.model == "mobi-resnet":
    model_test_keys = ['mobilenet', 'resnet50', 'resnet152']
elif args.model == "mobi-vgg":
    model_test_keys = ['mobilenet', 'vgg19_bn']
else:
    raise  ValueError("Model not found: %s" % (args.model))

# -------------  Train Temperature --------------
logger.info("**** Train Temperature ****")

for key in model_test_keys:
    logger.info("Train Temperature %s", key)
    # print(models[key](list(train_dataloader)[0]))
    # for data in tqdm(train_dataloader, desc="Evaluating"):
    #     with torch.no_grad():
    #         image = data[0].to(device)
    #         label = data[1].to(device)
    #         print(models[key](image))
    #         break
    models[key] = ModelWithTemperature(models[key])
    models[key].set_logger(logger)
    models[key].set_temperature(train_dataloader)

# -------------  Evaluation WITH Temperature --------------
logger.info("**** Evaluation WITH Temperature ****")

correct_cnt = dict(zip(model_keys, [0]*len(model_keys)))
coop_cnt = dict(zip(model_keys, [0]*len(model_keys)))
process_cnt = dict(zip(model_keys, [0]*len(model_keys)))

# num_batch = 0

num_labels = 0
th_stats = []
threshold = None
for data in tqdm(test_dataloader, desc="Evaluating"):
    image = data[0].to(device)
    label = data[1].to(device)

    if num_labels < 20 :
        with torch.no_grad():
            logits = models[model_test_keys[0]](image)
            probabilities = m(logits).cpu()
            th_stats.append(np.max(probabilities.cpu().detach().numpy(), axis=1).tolist())
        num_labels += eval_batch_size
        continue
    elif threshold is None:
        threshold = np.mean(th_stats)

    mask = np.array([False]*len(label.cpu()))
    for key in model_test_keys:
        with torch.no_grad():
            logits = models[key](image)
        probabilities = m(logits).cpu()
            
        model_ans = np.argmax(probabilities.cpu().detach().numpy(), axis=1)
        true_ans = label.cpu().detach().numpy().flatten()

        selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])

        processed = (selected_prob >= threshold) if key in model_test_keys[:-1] else np.array([True]*len(label.cpu()))
        correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
        coop_cnt[key] += np.count_nonzero((model_ans == true_ans) & (~mask) & processed)
        process_cnt[key] += np.count_nonzero((~mask) & processed)
        mask |= processed
    num_labels += len(label.cpu())

    # num_batch += 1
    # if num_batch > 10: break
    # if num_labels > 1000:
    #     break

num_labels -= len(np.array(th_stats).flatten())

logger.info("  Num examples = %s", num_labels)
logger.info("  Threshold = %s", threshold)
# for key in model_test_keys:
#     logger.info("final temperature %s", models[key].temperature)
logger.info("***** Eval results *****")
for key in model_test_keys:
    logger.info("%s correct count %s, percent %d", key, correct_cnt[key], int(correct_cnt[key]/num_labels * 100))
logger.info("***** Collaborative Eval results *****")
for key in model_test_keys:
    logger.info("%s process count %s, correct count %s, percent %d", key, process_cnt[key], coop_cnt[key], int(coop_cnt[key]/process_cnt[key] * 100))
