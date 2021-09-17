import torch
import deepspeed
import logging
import argparse
import subprocess
import os
import numpy as np

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, DistilBertForSequenceClassification

from data_processor import processors, bert_base_model_config, output_modes
from utils import convert_examples_to_features
from eval import compute_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


base_dir = "/home/oai/share"

def initialize():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.",
    )

    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--random",
        default=False,
        action="store_true",
        help="Whether to fientune for random initialization",
    )
    parser.add_argument(
        "--focal",
        default=False,
        action="store_true",
        help="Whether to use Focal Loss for finetuning.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma parameter to be used in focal loss.",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

args = initialize()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = args.task_name.lower()

processor = processors[task_name]()
output_mode = output_modes[task_name]

label_list = processor.get_labels()
num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(f"{base_dir}/HuggingFace/bert-base-uncased-{args.task_name}")

model_keys = [
    "S", 
    "M", 
    # "L",
]

model_paths = [
    f"{base_dir}/HuggingFace/distilbert-base-uncased-{args.task_name}",
    f"{base_dir}/HuggingFace/bert-base-uncased-{args.task_name}",
    # "/home/oai/efficient-nlp/outputs/bert-large-uncased/CoLAFineTune_bsz16_lr0.00003_epoch20_CoLA/finetuned_quantized_checkpoints/epoch4_step2675",
]

model_paths = dict(zip(model_keys, model_paths))

models = dict()
for key in model_keys:
    logger.debug("key %s, path %s", key, model_paths[key])
    models[key] = AutoModelForSequenceClassification.from_pretrained(model_paths[key]).to(device) # if key != "S" else DistilBertForSequenceClassification.from_pretrained(model_paths[key])

# -------------  Dataset Prepare --------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels != None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

dev_texts = processor.get_dev_tsv(args.data_dir)
encoded_dev_texts = tokenizer(dev_texts["sentence"].to_list(), padding = True, truncation = True, max_length=args.max_seq_length, return_tensors = 'pt')
eval_dataset = Dataset(encoded_dev_texts, torch.tensor(dev_texts['label']))
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(
    eval_dataset, sampler=eval_sampler, batch_size=1
)

num_labels = len(dev_texts['label'])

test_texts = processor.get_test_tsv(args.data_dir)
encoded_test_texts = tokenizer(test_texts["sentence"].to_list(), padding = True, truncation = True, max_length=args.max_seq_length, return_tensors = 'pt')
test_dataset = Dataset(encoded_test_texts)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(
    test_dataset, sampler=test_sampler, batch_size=1
)

m = torch.nn.Softmax(dim=1)

import pandas as pd

test_labels = dict()
for key in model_keys: test_labels[key] = list()

for batch in tqdm(test_dataloader, desc="Testing"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    for key in model_keys:
        with torch.no_grad():
            output = models[key](input_ids=input_ids, attention_mask=attention_mask, labels=None)
            logits = output.logits
        model_ans = np.argmax(m(output.logits).cpu(),axis=1)[0]
        if args.task_name == "RTE" or args.task_name == "QNLI":
            model_ans = "entailment" if model_ans == 0 else "not_entailment"
        test_labels[key].append(model_ans)

data_dir = f"/data/GlueData/{args.task_name}"
data_file = "test.tsv"
if args.task_name == 'MNLI-m':
    data_dir = data_dir.split('-')[0]
    data_file = "test_matched.tsv"
elif args.task_name == 'MNLI-mm':
    data_dir = data_dir.split('-')[0]
    data_file = "test_mismatched.tsv"
result = subprocess.run(['wc', '-l', os.path.join(data_dir, data_file)], stdout=subprocess.PIPE)
result = result.stdout
num_lines = int(result.split()[0])
ids = [x for x in range(num_lines)]
labels = ["entailment"]*num_lines if args.task_name in ['RTE', 'QNLI', 'MNLI-m', 'MNLI-mm'] else ["1"]*num_lines
test_result = pd.DataFrame({"id": ids, "label_tmp": labels})

for key in model_keys:
    # test_result = test_texts.copy()
    # test_result['label'] = test_labels[key]
    test_texts['label'] = test_labels[key]
    # print(test_texts.info(), test_texts)
    test_result = test_result.join(test_texts, on='id', how='left', rsuffix='_other')
    # print(test_result.info(), test_result)
    test_result = test_result.fillna("entailment" if args.task_name in ['RTE', 'QNLI', 'MNLI-m', 'MNLI-mm'] else "1")
    test_result = test_result[['id', 'label']]
    # print(test_result.info(), test_result)
    test_result.to_csv(f"TestResult/{args.task_name}-{key}.tsv", sep='\t',index=False)

# exit()

correct_cnt = dict(zip(model_keys, [0]*len(model_keys)))
coop_cnt = dict(zip(model_keys, [0]*len(model_keys)))

import time

# for input_ids, attention_mask, label_ids in tqdm(
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # print(input_ids[:64])

    flag = False
    for key in model_keys:
        with torch.no_grad():
            start = time.process_time()
            output = models[key](input_ids=input_ids, attention_mask=attention_mask, labels=None)
            end = time.process_time()
            logger.info("%s time elapsed: %s seconds", key, end-start)
            logits = output.logits

        model_ans = np.argmax(m(output.logits).cpu(),axis=1)[0]
        true_ans = labels[0].cpu()

        if model_ans == true_ans:
            correct_cnt[key] += 1
            coop_cnt[key] += 1 if not flag else 0
            flag = True
    
logger.info("***** Eval results *****")
logger.info("  Num examples = %d", num_labels)
logger.info("DistilBERT correct count %s, percent %d", correct_cnt['S'], int(correct_cnt['S']/num_labels * 100))
logger.info("BERTbase correct count %s, percent %d", correct_cnt['M'], int(correct_cnt['M']/num_labels * 100))

logger.info("***** Collaborative Eval results *****")
logger.info("DistilBERT correct count %s, percent %d", coop_cnt['S'], int(coop_cnt['S']/num_labels * 100))
logger.info("BERTbase correct count %s, percent %d", coop_cnt['M'], int(coop_cnt['M']/num_labels * 100))


#     # print(input_ids)
#     with torch.no_grad():
#         # print(input_ids.shape, input_mask.shape, segment_ids.shape, label_ids.shape)
#         output_base = model(
#             input_ids=input_ids, attention_mask=input_mask, labels=None)
#         logits = output_base.logits
#         print(input_ids.cpu()[:,-1] == 102, np.argmax(m(output_base.logits).cpu(),axis=1), label_ids)

#     if output_mode == "classification":
#             if args.focal:
#                 loss_fct = FocalLoss(class_num=num_labels, gamma=args.gamma)
#             else:
#                 loss_fct = CrossEntropyLoss()
#             tmp_eval_loss = loss_fct(
#                 logits.view(-1, num_labels), label_ids.view(-1)
#             )
#     elif output_mode == "regression":
#         loss_fct = MSELoss()
#         print(logits.type())
#         print(label_ids.type())
#         if task_name == "sts-b":
#             tmp_eval_loss = loss_fct(
#                 logits.float().view(-1), label_ids.view(-1)
#             )
#         else:
#             tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
#     eval_loss += tmp_eval_loss.mean().item()
#     nb_eval_steps += 1
#     if len(preds) == 0:
#         preds.append(logits.detach().cpu().numpy())
#     else:
#         preds[0] = np.append(
#             preds[0], logits.detach().cpu().numpy(), axis=0)

# # logger.debug("eval_loss %s, nb_eval_steps %s", eval_loss, nb_eval_steps)
# # logger.debug("tr_loss %s, nb_tr_steps %s", tr_loss, nb_tr_steps)
# eval_loss = eval_loss / nb_eval_steps
# preds = preds[0]
# if output_mode == "classification":
#     preds = np.argmax(preds, axis=1)
# elif output_mode == "regression":
#     preds = np.squeeze(preds)

# print(len(preds), len(all_label_ids))
# result = compute_metrics(task_name, preds, all_label_ids.numpy())

# result["eval_loss"] = eval_loss

# logger.info("***** Eval results *****")
# for key in sorted(result.keys()):
#     logger.info("  %s = %s", key, str(result[key]))
#     # if len(input_ids) < args.eval_batch_size: continue

#     input_ids = input_ids.to(device)
#     input_mask = input_mask.to(device)
#     segment_ids = segment_ids.to(device)
#     label_ids = label_ids.to(device)

#     # print(input_ids)
#     with torch.no_grad():
#         # print(input_ids.shape, input_mask.shape, segment_ids.shape, label_ids.shape)
#         output_base = model(
#             input_ids=input_ids, attention_mask=input_mask, labels=None)
#         logits = output_base.logits
#         print(input_ids.cpu()[:,-1] == 102, np.argmax(m(output_base.logits).cpu(),axis=1), label_ids)

#     if output_mode == "classification":
#             if args.focal:
#                 loss_fct = FocalLoss(class_num=num_labels, gamma=args.gamma)
#             else:
#                 loss_fct = CrossEntropyLoss()
#             tmp_eval_loss = loss_fct(
#                 logits.view(-1, num_labels), label_ids.view(-1)
#             )
#     elif output_mode == "regression":
#         loss_fct = MSELoss()
#         print(logits.type())
#         print(label_ids.type())
#         if task_name == "sts-b":
#             tmp_eval_loss = loss_fct(
#                 logits.float().view(-1), label_ids.view(-1)
#             )
#         else:
#             tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
#     eval_loss += tmp_eval_loss.mean().item()
#     nb_eval_steps += 1
#     if len(preds) == 0:
#         preds.append(logits.detach().cpu().numpy())
#     else:
#         preds[0] = np.append(
#             preds[0], logits.detach().cpu().numpy(), axis=0)

# # logger.debug("eval_loss %s, nb_eval_steps %s", eval_loss, nb_eval_steps)
# # logger.debug("tr_loss %s, nb_tr_steps %s", tr_loss, nb_tr_steps)
# eval_loss = eval_loss / nb_eval_steps
# preds = preds[0]
# if output_mode == "classification":
#     preds = np.argmax(preds, axis=1)
# elif output_mode == "regression":
#     preds = np.squeeze(preds)

# print(len(preds), len(all_label_ids))
# result = compute_metrics(task_name, preds, all_label_ids.numpy())

# result["eval_loss"] = eval_loss

# logger.info("***** Eval results *****")
# for key in sorted(result.keys()):
#     logger.info("  %s = %s", key, str(result[key]))




# model = AutoModelForSequenceClassification.from_pretrained(f"{base_dir}/HuggingFace/bert-base-uncased-{args.task_name}").to(device)
# model = BertForSequenceClassification.from_pretrained("/home/oai/efficient-nlp/outputs/bert-large-uncased/SST-2FineTune_bsz32_lr0.0005_epoch10_SST-2/checkpoint-8400").to(device)
# model = BertForSequenceClassification.from_pretrained("/home/oai/efficient-nlp/outputs/bert-large-uncased/CoLAFineTune_bsz16_lr0.00003_epoch10_CoLA/checkpoint-1000").to(device)
# model = AutoModelForSequenceClassification.from_pretrained(f"{base_dir}/HuggingFace/distilbert-base-uncased-{args.task_name}").to(device)


    