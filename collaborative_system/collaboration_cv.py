import torch
from torchvision import datasets, transforms
import torchvision.models as models

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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default=None,
    type=str,
    required=True,
    help="The type of model, e.g., resnet, vgg, incep-resnet, incep-vgg, mobi-resnet, mobi-vgg",
)

args = parser.parse_args()

eval_batch_size = 32

fh = logging.FileHandler(f'{__file__}_{args.model}.log', mode='w')
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

correct_cnt = dict(zip(model_keys, [0]*len(model_keys)))
coop_cnt = dict(zip(model_keys, [0]*len(model_keys)))


num_labels = 0
for data in tqdm(val_loader, desc="Evaluating"):
    image = data[0].to(device)
    label = data[1].to(device)

    mask = np.array([False]*len(label.cpu()))
    for key in model_test_keys:
        with torch.no_grad():
            output = models[key](image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        # print(np.argmax(probabilities), label)
        model_ans = np.argmax(probabilities.cpu().detach().numpy(), axis=1)
        true_ans = label.cpu().detach().numpy()

        # print(probabilities.cpu().shape, model_ans, true_ans)

        correct = (model_ans == true_ans)
        correct_cnt[key] += np.count_nonzero(correct)
        coop_cnt[key] += np.count_nonzero(correct & (~mask))
        # print(key, len(label), np.count_nonzero(correct), np.count_nonzero(correct & (~mask)))
        mask |= correct
    num_labels += eval_batch_size
    # if num_labels > 1000:
    #     break

logger.info("  Num examples = %d", num_labels)
logger.info("***** Eval results *****")
for key in model_test_keys:
    logger.info("%s correct count %s, percent %d", key, correct_cnt[key], int(correct_cnt[key]/num_labels * 100))
logger.info("***** Collaborative Eval results *****")
for key in model_test_keys:
    logger.info("%s correct count %s, percent %d", key, coop_cnt[key], int(coop_cnt[key]/num_labels * 100))
