import functools
import logging
import argparse
import os
import torch
import numpy as np
import deepspeed

import os

from ecosys.algo.monte_carlo import monte_carlo_bounds

# from transformers.models.auto.configuration_auto import AutoConfig
os.environ["NCCL_DEBUG"] = "DEBUG"
os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['NUMEXPR_MAX_THREADS'] = '48'
# os.environ["NCCL_SOCKET_IFNAME"] = "eno1"

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from transformers import AutoTokenizer, GPT2Tokenizer, GPT2ForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments

from ecosys.utils.data_structure import Dataset
from ecosys.utils.data_processor import processors, output_modes
from ecosys.utils.eval import compute_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


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
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="The name of the model to train.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
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
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


args = initialize()

# from accelerate import Accelerator
# accelerator = Accelerator()
# device = accelerator.device

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = args.task_name.lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
output_mode = output_modes[task_name]

# label_list = processor.get_labels()
# num_labels = len(label_list)
num_labels=1

model_name = f"/home/oai/share/HuggingFace/{args.model_name}/"

# tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=args.do_lower_case)

def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    if token_ids_1 != None:
        outputs += token_ids_1 + [self.eos_token_id]
    return outputs

GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
tokenizer = GPT2Tokenizer.from_pretrained(model_name, do_lower_case=args.do_lower_case)
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
tokenizer.pad_token = tokenizer.unk_token

# SPECIAL_TOKENS  = { "bos_token": "[CLS]",
#                     "eos_token": "[EOS]",
#                     "unk_token": "[UNK]",                    
#                     "pad_token": "[PAD]",
#                     "sep_token": "[SEP]"}
# tokenizer.add_special_tokens(SPECIAL_TOKENS)
config = AutoConfig.from_pretrained(model_name, num_labels=1)
# config = AutoConfig.from_pretrained(model_name, 
#                                     bos_token_id=tokenizer.bos_token_id,
#                                     eos_token_id=tokenizer.eos_token_id,
#                                     sep_token_id=tokenizer.sep_token_id,
#                                     pad_token_id=tokenizer.pad_token_id,
#                                     # output_hidden_states=False
#                                     num_labels=num_labels,
#                                 )

# tokenizer.pad_token = tokenizer.eos_token
# config.pad_token_id = config.eos_token_id

# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)

training_args = TrainingArguments(
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=32,  # batch size per device during training
    weight_decay=0.01,               # strength of weight decay
    load_best_model_at_end=True,
    logging_strategy='epoch',
    save_strategy='epoch',
    # save_total_limit=1,
    evaluation_strategy="epoch",
    output_dir=args.output_dir,
    learning_rate=6.25e-5,
    # dataloader_drop_last=True,
    fp16=True,
    fp16_opt_level="O3",
    fp16_full_eval=True,
    warmup_steps=1e2,
)

# -------------  Dataset Prepare --------------

train_texts = processor.get_train_tsv(args.data_dir)
encoded_train_texts = tokenizer(
    train_texts["sentence"].to_list(),
    padding = 'max_length', 
    truncation = True, 
    max_length=args.max_seq_length, 
    return_tensors = 'pt'
)

for key in encoded_train_texts:
    logger.debug("---%s--- %s", key, encoded_train_texts[key].shape)

# for t in encoded_train_texts['input_ids']:
#     logger.debug("===== %s %s", t, t.shape)

dev_texts = processor.get_dev_tsv(args.data_dir)
encoded_dev_texts = tokenizer(
    dev_texts["sentence"].to_list(),
    padding = 'max_length', 
    truncation = True, 
    max_length=args.max_seq_length,
    return_tensors = 'pt'
)

# print(train_texts.info())
# print(train_texts.head())

train_dataset = Dataset(encoded_train_texts, torch.tensor(train_texts['label']))
eval_dataset = Dataset(encoded_dev_texts, torch.tensor(dev_texts['label']))

model = GPT2ForSequenceClassification.from_pretrained(model_name, config=config).cuda()
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = -100

print(model.config.num_labels)
# exit()


def compute_acc(predictions, labels, th):
    predictions = np.where(predictions > th[0], 1, 0)
    return compute_metrics(args.task_name.lower(), predictions, labels)['mcc']

def metrics(eval_pred):
    print(eval_pred)
    # predictions, labels = eval_pred
    predictions = eval_pred.predictions.reshape((-1, num_labels))
    labels = eval_pred.label_ids.flatten()

    # threshold_bounds = monte_carlo_bounds(
    #     functools.partial(compute_acc, predictions, labels), 
    #     [(np.min(predictions), np.max(predictions))], 
    #     [('reward', float)],
    #     n=1000,
    #     tops=20,
    #     maxiter=20,
    # )
    # logger.info("------threshold %s -----", np.mean(threshold_bounds))
    # predictions = np.where(predictions > np.mean(threshold_bounds), 1, 0)
    predictions = np.where(predictions > 0.5, 1, 0)
    # predictions = np.argmax(predictions, axis=1)
    # print(len(predictions), np.count_nonzero(predictions), np.count_nonzero(labels))
    return compute_metrics(args.task_name.lower(), predictions, labels)

# -------------  Model Training --------------

# def model_init():
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True).cuda()
#     model.resize_token_embeddings(len(tokenizer))
#     return model


# model.config.num_labels = num_labels
# model.transformer.eval()

# for parameter in model.transformer.parameters():
#     parameter.requires_grad = False

# for i, m in enumerate(model.transformer.h):        
#     #Only un-freeze the last n transformer blocks
#     if i+1 > 6:
#         for parameter in m.parameters():
#             parameter.requires_grad = True 

# for parameter in model.transformer.ln_f.parameters():        
#     parameter.requires_grad = True

# for parameter in model.transformer.lm_head.parameters():        
#     parameter.requires_grad = True

# def BERT_hp_space(trial):
#     return {
#         "learning_rate": trial.suggest_categorical("learning_rate", [5e-5, 3e-5, 2e-5]),
#         "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.3, log=True),
#         "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10, log=True),
#         "seed": trial.suggest_int("seed", 20, 40, log=True),
#         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
#     }

trainer = Trainer(
    model = model,
    # model_init=model_init,
    compute_metrics=metrics,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR)
)
# trainer.hyperparameter_search(direction="maximize", n_trials=20, hp_space=BERT_hp_space, backend="optuna")
trainer.train()

