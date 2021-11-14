import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification
from ecosys.context.arg_parser import ArgParser
from ecosys.utils.data_structure import HuggingFaceDataset
from network_attached_model import NetworkAttachedModel
from ecosys.context.srv_ctx import ServiceContext
from ecosys.utils.data_processor import processors, output_modes

import threading
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

args = ArgParser().parse()
ctx = ServiceContext(args.cfg)

sequence_length = 128
task_name = 'SST-2'

batch_size = ctx.cfg.model_bsz

base_dir = "/home/oai/share"
tokenizer = AutoTokenizer.from_pretrained(
    f"{base_dir}/HuggingFace/bert-base-uncased-{task_name}")

# -------------  Dataset Prepare --------------

processor = processors[task_name.lower()]()
output_mode = output_modes[task_name.lower()]


def data_preprocessing():
    # train = processor.get_train_tsv(
    #     f'/data/GlueData/{task_name}/').reset_index()
    texts = processor.get_dev_tsv(f'/data/GlueData/{task_name}/').reset_index()
    train, test = train_test_split(texts, test_size=0.5, random_state=0)

    encoded_texts = tokenizer(
        train["sentence"].to_list(),
        padding='max_length',
        truncation=True,
        max_length=sequence_length,
        return_tensors='pt'
    )
    dataset = HuggingFaceDataset(encoded_texts, torch.tensor(
        train['label'].to_list()), ctx.cfg.srv_device)
    sampler = SequentialSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    encoded_texts = tokenizer(
        test["sentence"].to_list(),
        padding='max_length',
        truncation=True,
        max_length=sequence_length,
        return_tensors='pt'
    )
    dataset = HuggingFaceDataset(encoded_texts, torch.tensor(
        test['label'].to_list()), ctx.cfg.srv_device)
    sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )

    return train_dataloader, test_dataloader


m = torch.nn.Softmax(dim=1)

train_dataloader, test_dataloader = data_preprocessing()


if __name__ == "__main__":
    model = NetworkAttachedModel(
        AutoModelForSequenceClassification.from_pretrained(ctx.cfg.model_path, return_dict=True),
        ctx
    )
    model.monitor()
    model.prepare(train_dataloader)
    model.serve(True)
    
