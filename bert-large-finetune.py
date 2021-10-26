import logging
import argparse
import os
import torch
import numpy as np
import deepspeed

import os
os.environ["NCCL_DEBUG"] = "DEBUG"
# os.environ["NCCL_SOCKET_IFNAME"] = "eno1"

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
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

label_list = processor.get_labels()
num_labels = len(label_list)

model_name = f"/home/oai/share/HuggingFace/{args.model_name}/"

tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=args.do_lower_case)

# train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)

training_args = TrainingArguments(
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=32,  # batch size per device during training
    weight_decay=0.01,               # strength of weight decay
    load_best_model_at_end=True,
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    evaluation_strategy="epoch",
    output_dir=args.output_dir,
    learning_rate=3e-5,
    dataloader_drop_last=True,
    fp16=True,
    fp16_opt_level="O3",
    fp16_full_eval=True,
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
    logger.debug("--------------------%s--- %s", key, encoded_train_texts[key].shape)

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

def metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    print(len(predictions), np.count_nonzero(predictions), np.count_nonzero(labels))
    return compute_metrics(args.task_name.lower(), predictions, labels)

# -------------  Model Training --------------

def model_init():
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True)
    # model.half()
    return model

from deepspeed.pipe import PipelineModule
# net = PipelineModule(layers=net, num_stages=2)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, return_dict=True)
model.base_model = PipelineModule(layers=model.base_model.modules(), num_stages=2)

def BERT_hp_space(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [5e-5, 3e-5, 2e-5]),
        "weight_decay": trial.suggest_float("weight_decay", 0.1, 0.3, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4, log=True),
        "seed": trial.suggest_int("seed", 20, 40, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
    }

trainer = Trainer(
    model = model,
    # model_init=model_init,
    compute_metrics=metrics,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # optimizers=(optimizer, torch.optim.lr_scheduler.LambdaLR)
)
# trainer.hyperparameter_search(direction="maximize", n_trials=20, hp_space=BERT_hp_space)
trainer.train()

# # eval_dataset = TensorDataset(encoded_texts)
# eval_sampler = SequentialSampler(eval_dataset)
# eval_dataloader = DataLoader(
#     eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
# )

# logger.info("***** Running evaluation *****")
# logger.info("  Num examples = %d", len(encoded_texts['labels']))
# logger.info("  Batch size = %d", args.eval_batch_size)

# model.eval()
# eval_loss = 0
# nb_eval_steps = 0
# preds = []

# for input_ids, attention_mask, labels in tqdm(
#     eval_dataloader, desc="Evaluating"
# ):
#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     labels = labels.to(device)

#     with torch.no_grad():
#         logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=None).logits

#     # create eval loss and other metric required by the task
#     if output_mode == "classification":
#         loss_fct = CrossEntropyLoss()
#         tmp_eval_loss = loss_fct(
#             logits.view(-1, num_labels), label_ids.view(-1)
#         )
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
#     logger.debug("tmp_eval_loss %s", tmp_eval_loss.mean().item())
#     eval_loss += tmp_eval_loss.mean().item()
#     nb_eval_steps += 1
#     if len(preds) == 0:
#         preds.append(logits.detach().cpu().numpy())
#     else:
#         preds[0] = np.append(
#             preds[0], logits.detach().cpu().numpy(), axis=0)

# logger.debug("eval_loss %s, nb_eval_steps %s", eval_loss, nb_eval_steps)
# logger.debug("tr_loss %s, nb_tr_steps %s", tr_loss, nb_tr_steps)
# eval_loss = eval_loss / nb_eval_steps
# preds = preds[0]
# if output_mode == "classification":
#     preds = np.argmax(preds, axis=1)
# elif output_mode == "regression":
#     preds = np.squeeze(preds)
# result = compute_metrics(task_name, preds, all_label_ids.numpy())
# loss = tr_loss / nb_tr_steps if args.do_train else None

# result["eval_loss"] = eval_loss
# result["global_step"] = global_step
# result["loss"] = loss

# output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
# with open(output_eval_file, "w") as writer:
#     logger.info("***** Eval results *****")
#     for key in sorted(result.keys()):
#         logger.info("  %s = %s", key, str(result[key]))
#         writer.write("%s = %s\n" % (key, str(result[key])))

# # hack for MNLI-MM
# if task_name == "mnli":
#     task_name = "mnli-mm"
#     processor = processors[task_name]()

#     if (
#         os.path.exists(args.output_dir + "-MM")
#         and os.listdir(args.output_dir + "-MM")
#         and args.do_train
#     ):
#         raise ValueError(
#             "Output directory ({}{}) already exists and is not empty.".format(
#                 args.output_dir, "-MM"
#             )
#         )
#     if not os.path.exists(args.output_dir + "-MM"):
#         os.makedirs(args.output_dir + "-MM")

#     eval_examples = processor.get_dev_examples(args.data_dir)
#     eval_features = convert_examples_to_features(
#         eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
#     )
#     logger.info("***** Running evaluation *****")
#     logger.info("  Num examples = %d", len(eval_examples))
#     logger.info("  Batch size = %d", args.eval_batch_size)
#     all_input_ids = torch.tensor(
#         [f.input_ids for f in eval_features], dtype=torch.long
#     )
#     all_input_mask = torch.tensor(
#         [f.input_mask for f in eval_features], dtype=torch.long
#     )
#     all_segment_ids = torch.tensor(
#         [f.segment_ids for f in eval_features], dtype=torch.long
#     )
#     all_label_ids = torch.tensor(
#         [f.label_id for f in eval_features], dtype=torch.long
#     )

#     eval_data = TensorDataset(
#         all_input_ids, all_input_mask, all_segment_ids, all_label_ids
#     )
#     # Run prediction for full data
#     eval_sampler = SequentialSampler(eval_data)
#     eval_dataloader = DataLoader(
#         eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
#     )

#     model.eval()
#     eval_loss = 0
#     nb_eval_steps = 0
#     preds = []

#     for input_ids, input_mask, segment_ids, label_ids in tqdm(
#         eval_dataloader, desc="Evaluating"
#     ):
#         input_ids = input_ids.to(device)
#         input_mask = input_mask.to(device)
#         segment_ids = segment_ids.to(device)
#         label_ids = label_ids.to(device)

#         with torch.no_grad():
#             logits = model(input_ids, segment_ids, input_mask, labels=None)

#         if args.focal:
#             loss_fct = FocalLoss(class_num=num_labels, gamma=args.gamma)
#         else:
#             loss_fct = CrossEntropyLoss()
#         tmp_eval_loss = loss_fct(
#             logits.view(-1, num_labels), label_ids.view(-1)
#         )
#         logger.debug("tmp_eval_loss %s", tmp_eval_loss)
#         logger.debug("logits %s %s %s, label_ids %s", logits,
#                         logits.view(-1), logits.view(-1, num_labels), label_ids.view(-1))
#         eval_loss += tmp_eval_loss.mean().item()
#         nb_eval_steps += 1
#         if len(preds) == 0:
#             preds.append(logits.detach().cpu().numpy())
#         else:
#             preds[0] = np.append(
#                 preds[0], logits.detach().cpu().numpy(), axis=0
#             )
#     logger.debug("eval_loss %s, nb_eval_steps %s",
#                     eval_loss, nb_eval_steps)
#     logger.debug("tr_loss %s, nb_tr_steps %s", tr_loss, nb_tr_steps)
#     eval_loss = eval_loss / nb_eval_steps
#     preds = preds[0]
#     preds = np.argmax(preds, axis=1)
#     result = compute_metrics(task_name, preds, all_label_ids.numpy())
#     loss = tr_loss / nb_tr_steps if args.do_train else None

#     result["eval_loss"] = eval_loss
#     result["global_step"] = global_step
#     result["loss"] = loss

#     output_eval_file = os.path.join(
#         args.output_dir + "-MM", "eval_results.txt")
#     with open(output_eval_file, "w") as writer:
#         logger.info("***** Eval results *****")
#         for key in sorted(result.keys()):
#             logger.info("  %s = %s", key, str(result[key]))
#             writer.write("%s = %s\n" % (key, str(result[key])))
