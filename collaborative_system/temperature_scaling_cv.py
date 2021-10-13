import torch
from torchvision import datasets, transforms
import torchvision.models as models
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from ecosys.utils.data_structure import TorchVisionDataset
from ecosys.models.temperature_scaling import ModelWithTemperature
from ecosys.algo.monte_carlo import monte_carlo_bounds
from ecosys.decorators.model_decorators import model_eval

from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import ViTFeatureExtractor, ViTForImageClassification



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

fh = logging.FileHandler(f'{filename}_{args.model}.log', mode='a')
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


def data_preprocessing():
    val_dataset = TorchVisionDataset(datasets.ImageNet("/home/oai/share/dataset/.", split="val", transform=preprocess))

    index = np.array([x for x in range(len(val_dataset))])
    train_index, test_index = train_test_split(index, test_size=0.6)
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

model_names = [
    'resnet18',
    'resnet50',
    'resnet152',
    'vgg11',
    'vgg16',
    'vgg19_bn',
    'inception',
    'mobilenet',
    'ViT-base',
    'ViT-large',
]

energy_discount_factor = [
    0.25, 
    0.5, 
    1.0,
    0.25,
    1.6,
    2.0,
    0.375,
    0.075,
    1.5,
    4.0,
]

model_energy = dict(zip(model_names, energy_discount_factor))

tokenizer = ViTFeatureExtractor.from_pretrained()

model_instances = [
    models.resnet18(pretrained=True),
    models.resnet50(pretrained=True),
    models.resnet152(pretrained=True),
    models.vgg11(pretrained=True),
    models.vgg16(pretrained=True),
    models.vgg19_bn(pretrained=True),
    models.inception_v3(pretrained=True),
    models.mobilenet_v2(pretrained=True),
    ViTForImageClassification.from_pretrained('/home/oai/share/HuggingFace/vit-base-patch32-384/'),
    ViTForImageClassification.from_pretrained('/home/oai/share/HuggingFace/vit-large-patch32-384/'),
]

for model in model_instances:
    model.to(device)
    model.eval()

models = dict(zip(model_names, model_instances))

# resnet18 = models.resnet18(pretrained=True)
# resnet50 = models.resnet50(pretrained=True)
# resnet152 = models.resnet152(pretrained=True)
# vgg11 = models.vgg11(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# vgg19_bn = models.vgg19_bn(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# mobilenet = models.mobilenet_v2(pretrained=True)

if args.model == "resnet":
    model_keys = ['resnet18', 'resnet50', 'resnet152']
elif args.model == "vgg":
    model_keys = ['vgg11', 'vgg16', 'vgg19_bn']
elif args.model == "incep-resnet":
    model_keys = ['inception', 'resnet50', 'resnet152']
elif args.model == "incep-vgg":
    model_keys = ['inception', 'vgg19_bn']
elif args.model == "mobi-resnet":
    model_keys = ['mobilenet', 'resnet50', 'resnet152']
elif args.model == "mobi-vgg":
    model_keys = ['mobilenet', 'vgg19_bn']
elif args.model == "vit":
    model_keys = ['ViT-base', 'ViT-large']
else:
    raise  ValueError("Model not found: %s" % (args.model))

# -------------  Train Temperature --------------
logger.info("**** Train Temperature ****")

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

threshold_bounds = monte_carlo_bounds(
        total_reward, 
        [(0.5, 1.0),(0.5, 1.0)], 
        [('reward', float), ('energy', float)],
        n=1000,
        tops=20,
        maxiter=20,
    )
mc_threshold = np.mean(
    threshold_bounds, axis=1
)
logger.info("Threshold Bounds %s", threshold_bounds)
# -------------  Evaluation WITH Temperature --------------
logger.info("**** Evaluation WITH Temperature ****")

correct_cnt = dict(zip(model_names, [0]*len(model_names)))
correct_prob = dict(zip(model_keys, [0]*len(model_keys)))
coop_cnt = dict(zip(model_names, [0]*len(model_names)))
process_cnt = dict(zip(model_names, [0]*len(model_names)))
process_prob = dict(zip(model_keys, [0]*len(model_keys)))

# num_batch = 0

correct_cnt = dict(zip(model_keys, [0]*n_models))
correct_prob = dict(zip(model_keys, [0]*n_models))
coop_cnt = dict(zip(model_keys, [0]*n_models))
process_prob = dict(zip(model_keys, [0]*n_models))
process_cnt = dict(zip(model_keys, [0]*n_models))

num_labels = 0
# th_stats = []
# threshold = None

th_stats = dict(zip(model_keys, [list() for _ in range(n_models)]))

@model_eval(test_dataloader)
def eval_monte_carlo(input, label):

    global num_labels
    # global th_stats

    b_size = len(label.cpu())
    mask = np.array([False]*b_size)

    for i, key in enumerate(model_keys):
        logits = models[key](input)
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

eval_monte_carlo()


# for data in tqdm(test_dataloader, desc="Evaluating"):
#     image = data[0].to(device)
#     label = data[1].to(device)

#     if num_labels < 20 :
#         with torch.no_grad():
#             logits = models[model_keys[0]](image)
#             probabilities = m(logits).cpu()
#             th_stats.append(np.max(probabilities.cpu().detach().numpy(), axis=1).tolist())
#         num_labels += eval_batch_size
#         continue
#     elif threshold is None:
#         threshold = np.mean(th_stats)

#     mask = np.array([False]*len(label.cpu()))
#     for key in model_keys:
#         with torch.no_grad():
#             logits = models[key](image)
#         probabilities = m(logits).cpu()
            
#         model_ans = np.argmax(probabilities.cpu().detach().numpy(), axis=1)
#         true_ans = label.cpu().detach().numpy().flatten()

#         selected_prob = np.array([p[model_ans[i]] for i, p in enumerate(probabilities)])

#         processed = (selected_prob >= threshold) if key in model_keys[:-1] else np.array([True]*len(label.cpu()))
#         correct_cnt[key] += np.count_nonzero(model_ans == true_ans)
#         coop_cnt[key] += np.count_nonzero((model_ans == true_ans) & (~mask) & processed)
#         process_cnt[key] += np.count_nonzero((~mask) & processed)
#         mask |= processed
#     num_labels += len(label.cpu())

#     # num_batch += 1
#     # if num_batch > 10: break
#     # if num_labels > 1000:
#     #     break

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
plt.savefig(f'figures/th_stats_{args.model}.png', bbox_inches="tight")

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
