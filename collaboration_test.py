import transformers
import deepspeed
import logging
import argparse
import os
import torch
import random
import numpy as np

from tqdm import tqdm, trange
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from data_processor import processors, bert_base_model_config, output_modes
from utils import convert_examples_to_features, checkpoint_model
from eval import compute_metrics

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

def inference_wrapper(model):
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model)
    return model_engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_name = args.task_name.lower()

processor = processors[task_name]()
output_mode = output_modes[task_name]

label_list = processor.get_labels()
num_labels = len(label_list)

bert_config = BertConfig(**bert_base_model_config)
tokenizer = BertTokenizer.from_pretrained(
    "/home/oai/HuggingFace/bert-large-uncased/")
model_base = inference_wrapper(BertForSequenceClassification(bert_config))
model_distill = inference_wrapper(BertForSequenceClassification(bert_config))
# model_distill = inference_wrapper(DistilBertForSequenceClassification.from_pretrained("/home/oai/HuggingFace/distilbert-base-uncased-finetuned-sst-2-english"))

model_base.load_checkpoint(
    # "/home/oai/efficient-nlp/outputs/bert-base-uncased/CoLAFineTune_bsz32_lr0.00003_epoch4_CoLA/finetuned_quantized_checkpoints/"
    "/home/oai/BERT-checkpoints/bert_pyt_ckpt_base_ft_sst2_amp_128_20.12.0/"
)
model_distill.load_checkpoint(
    "/home/oai/efficient-nlp/outputs/distilbert-base-uncased/CoLAFineTune_bsz32_lr0.00002_epoch4_CoLA/finetuned_quantized_checkpoints/")
    # "/home/oai/efficient-nlp/outputs/distilbert-base-uncased/CoLAFineTune_bsz32_lr0.00002_epoch4_CoLA/finetuned_quantized_checkpoints/")
# model_base.from_pretrained(
#     "outputs/bert-base-uncased/CoLAFineTune_bsz32_lr0.00003_epoch4_CoLA")
# model_distill = BertForSequenceClassification.from_pretrained(
#     "outputs/distilbert-base-uncased/CoLAFineTune_bsz32_lr0.00003_epoch2_CoLA")


eval_examples = processor.get_dev_examples(args.data_dir)
eval_features = convert_examples_to_features(
    eval_examples, label_list, args.max_seq_length, tokenizer, output_mode
)
logger.info("***** Running evaluation *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", args.eval_batch_size)
all_input_ids = torch.tensor(
    [f.input_ids for f in eval_features], dtype=torch.long
)
all_input_mask = torch.tensor(
    [f.input_mask for f in eval_features], dtype=torch.long
)
all_segment_ids = torch.tensor(
    [f.segment_ids for f in eval_features], dtype=torch.long
)

if output_mode == "classification":
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.long
    )
elif output_mode == "regression":
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.float
    )

eval_data = TensorDataset(
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids
)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(
    eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
)

model_base.eval()
model_distill.eval()
eval_loss = 0
nb_eval_steps = 0
preds = []

m = torch.nn.Softmax(dim=1)

for input_ids, input_mask, segment_ids, label_ids in tqdm(
    eval_dataloader, desc="Evaluating"
):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    # print(input_ids)

    with torch.no_grad():
        output_base = model_base(
            input_ids, segment_ids, input_mask, labels=None)
        output_distill = model_distill(
            input_ids, segment_ids, input_mask, labels=None)
        # print(m(output_base.logits), m(output_distill.logits), label_ids)
        print(np.argmax(m(output_base.logits).cpu(),axis=1), np.argmax(m(output_distill.logits).cpu(),axis=1), label_ids)

#     # create eval loss and other metric required by the task
#     if output_mode == "classification":
#         if args.focal:
#             loss_fct = FocalLoss(class_num=num_labels, gamma=args.gamma)
#         else:
#             loss_fct = CrossEntropyLoss()
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
#     logger.debug("tmp_eval_loss %s, mean %s", tmp_eval_loss.mean().item())
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
