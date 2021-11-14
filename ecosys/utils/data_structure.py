import torch
import csv
from collections import deque
import threading

def queue2list(queue):
    data = []
    try:
        while queue.qsize() > 0: 
            data.append(queue.get())
    finally:
        return data


class ThreadSafeQueue():
    def __init__(self) -> None:
        self.queue = list()
        self.lock = threading.Lock()

    def __iter__(self):
        for val in self.queue:
            with self.lock:
                yield val

    def __getitem__(self, idx):
        with self.lock:
            return self.queue[idx]

    def __len__(self):
        with self.lock:
            return len(self.dataset)

    def append(self, val):
        with self.lock:
            self.queue.append(val)

    def clear(self):
        with self.lock:
            self.queue.clear()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class ViTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_extractor, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.device = device

    def __getitem__(self, idx):
        # print(self.dataset[idx][0])
        encodings = self.feature_extractor(self.dataset[idx][0], return_tensors='pt')   
        # print("bbbbb", encodings['pixel_values'].shape)
        # print("aaaa", encodings['pixel_values'].squeeze(0).shape)
        encodings['pixel_values'] = encodings['pixel_values'].squeeze(0).to(self.device)
        return (encodings, torch.tensor(self.dataset[idx][1], dtype=torch.long).to(self.device))

    def __len__(self):
        return len(self.dataset)

class TorchVisionDataset(torch.utils.data.Dataset):
    def __init__(self, input, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.input = input
        self.device = device

    def __getitem__(self, idx):
        item = {'x': self.input[idx][0].clone().detach().to(self.device)}
        return (item, self.input[idx][1] if self.input[idx][1] != None else None)

    def __len__(self):
        return len(self.input)

class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.encodings = encodings
        self.labels = labels
        self.device = device

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings['input_ids'][idx].clone().detach().to(self.device),
            "attention_mask": self.encodings['attention_mask'][idx].clone().detach().to(self.device),
        }
        return (item, torch.tensor([self.labels[idx].clone().detach()]).to(self.device) if self.labels != None else torch.tensor([0]*len(item['input_ids'])).to(self.device))

    def __len__(self):
        return len(self.encodings["input_ids"])

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, samples, labels=None):
#         self.samples = samples
#         self.labels = labels

#     def __getitem__(self, idx):
#         raise NotImplementedError()

#     def __len__(self):
#         raise NotImplementedError()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.encodings = encodings
        self.labels = labels
        self.device = device

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels != None:
            item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines