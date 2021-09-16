import transformers
import deepspeed
import logging
import argparse
import os
import torch
import random
import numpy as np

import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "virbr0"

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertConfig, BertModel, BertForSequenceClassification, DistilBertForSequenceClassification

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import warmup_linear

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
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.",
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
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help="The directory for downloaded pretained model",
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
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=8, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
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
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
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
    parser.add_argument(
        "--deepspeed_sparse_attention",
        default=False,
        action="store_true",
        help="Use DeepSpeed sparse self attention.",
    )
    parser.add_argument(
        "--preln",
        action="store_true",
        default=False,
        help="Switching to the variant of Transformer blocks that use pre-LayerNorm.",
    )
    parser.add_argument(
        "--deepspeed_transformer_kernel",
        default=False,
        action="store_true",
        help="Use DeepSpeed transformer kernel to accelerate.",
    )
    parser.add_argument(
        "--progressive_layer_drop",
        default=False,
        action="store_true",
        help="Whether to enable progressive layer dropping or not",
    )
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


args = initialize()

if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 2
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
logger.info(
    "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16
    )
)

if args.gradient_accumulation_steps < 1:
    raise ValueError(
        "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps
        )
    )

args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

args.seed = random.randint(1, 1000)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

task_name = args.task_name.lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
output_mode = output_modes[task_name]

label_list = processor.get_labels()
num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained(os.path.join(args.model_dir))
model = BertForSequenceClassification.from_pretrained(os.path.join(args.model_dir)).to(device)
bert_config = BertConfig(**bert_base_model_config)

# logger.debug("model %s", model)
# exit()

train_examples = None
num_train_optimization_steps = None
if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = (
        int(
            len(train_examples)
            / args.train_batch_size
            / args.gradient_accumulation_steps
        )
        * args.num_train_epochs
    )
    if args.local_rank != -1:
        num_train_optimization_steps = (
            num_train_optimization_steps // torch.distributed.get_world_size()
        )

# Prepare model
cache_dir = (
    args.cache_dir
    if args.cache_dir
    else os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(args.local_rank)
    )
)

def init_bert_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        print('random init')
        module.weight.data.normal_(mean=0.0, std=0.02/ np.sqrt(2.0 * 12))
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

bert_config = BertConfig(**bert_base_model_config)
bert_config.vocab_size = len(tokenizer.vocab)
# Padding for divisibility by 8
if bert_config.vocab_size % 8 != 0:
    bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)

if args.random:
    logger.info("USING RANDOM INITIALISATION FOR FINETUNING")
    model.apply(init_bert_weights)

# if args.fp16:
#     model.half()
model.to(device)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.01,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

model, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=optimizer_grouped_parameters,
    dist_init_required=True,
)

global_step = 0
nb_tr_steps = 0
tr_loss = 0
if args.do_train:
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, output_mode
    )
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long
    )

    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in train_features], dtype=torch.long
        )
    elif output_mode == "regression":
        if args.fp16:
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.half
            )
        else:
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.float
            )

    train_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    model.train()
    nb_tr_examples = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids = input_ids, attention_mask=input_mask, labels=None).logits

            if output_mode == "classification":
                if args.focal:
                    loss_fct = FocalLoss(
                        class_num=num_labels, gamma=args.gamma)
                else:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels),
                                label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.deepscale and args.local_rank != -1:
                model.disable_need_reduction()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    model.enable_need_reduction()

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            logger.debug("loss %s", loss.item())

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(
                        global_step / num_train_optimization_steps,
                        args.warmup_proportion,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        saved_path = os.path.join(
            args.output_dir, "finetuned_quantized_checkpoints")

        checkpoint_model(
            PATH=saved_path,
            ckpt_id="epoch{}_step{}".format(epoch, global_step),
            model=model,
            epoch=epoch,
            last_global_step=global_step,
            last_global_data_samples=nb_tr_examples
            * torch.distributed.get_world_size(),
        )
        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
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

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(
                eval_dataloader, desc="Evaluating"
            ):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None).logits

                    m = torch.nn.Softmax(dim=1)
                    print(np.argmax(m(logits).cpu(),axis=1), label_ids)

                # create eval loss and other metric required by the task
                if output_mode == "classification":
                    if args.focal:
                        loss_fct = FocalLoss(class_num=num_labels, gamma=args.gamma)
                    else:
                        loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(
                        logits.view(-1, num_labels), label_ids.view(-1)
                    )
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    print(logits.type())
                    print(label_ids.type())
                    if task_name == "sts-b":
                        tmp_eval_loss = loss_fct(
                            logits.float().view(-1), label_ids.view(-1)
                        )
                    else:
                        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
                # logger.debug("tmp_eval_loss %s", tmp_eval_loss.mean().item())
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            # logger.debug("eval_loss %s, nb_eval_steps %s", eval_loss, nb_eval_steps)
            # logger.debug("tr_loss %s, nb_tr_steps %s", tr_loss, nb_tr_steps)
            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(task_name, preds, all_label_ids.numpy())
            loss = tr_loss / nb_tr_steps if args.do_train else None

            result["eval_loss"] = eval_loss
            result["global_step"] = global_step
            result["loss"] = loss

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            # hack for MNLI-MM
            if task_name == "mnli":
                task_name = "mnli-mm"
                processor = processors[task_name]()

                if (
                    os.path.exists(args.output_dir + "-MM")
                    and os.listdir(args.output_dir + "-MM")
                    and args.do_train
                ):
                    raise ValueError(
                        "Output directory ({}{}) already exists and is not empty.".format(
                            args.output_dir, "-MM"
                        )
                    )
                if not os.path.exists(args.output_dir + "-MM"):
                    os.makedirs(args.output_dir + "-MM")

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
                all_label_ids = torch.tensor(
                    [f.label_id for f in eval_features], dtype=torch.long
                )

                eval_data = TensorDataset(
                    all_input_ids, all_input_mask, all_segment_ids, all_label_ids
                )
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(
                    eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size
                )

                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                preds = []

                for input_ids, input_mask, segment_ids, label_ids in tqdm(
                    eval_dataloader, desc="Evaluating"
                ):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask, labels=None)

                    if args.focal:
                        loss_fct = FocalLoss(class_num=num_labels, gamma=args.gamma)
                    else:
                        loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(
                        logits.view(-1, num_labels), label_ids.view(-1)
                    )
                    logger.debug("tmp_eval_loss %s", tmp_eval_loss)
                    logger.debug("logits %s %s %s, label_ids %s", logits,
                                logits.view(-1), logits.view(-1, num_labels), label_ids.view(-1))
                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(
                            preds[0], logits.detach().cpu().numpy(), axis=0
                        )
                logger.debug("eval_loss %s, nb_eval_steps %s",
                            eval_loss, nb_eval_steps)
                logger.debug("tr_loss %s, nb_tr_steps %s", tr_loss, nb_tr_steps)
                eval_loss = eval_loss / nb_eval_steps
                preds = preds[0]
                preds = np.argmax(preds, axis=1)
                result = compute_metrics(task_name, preds, all_label_ids.numpy())
                loss = tr_loss / nb_tr_steps if args.do_train else None

                result["eval_loss"] = eval_loss
                result["global_step"] = global_step
                result["loss"] = loss

                output_eval_file = os.path.join(
                    args.output_dir + "-MM", "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
