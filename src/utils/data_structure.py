import torch
import csv

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

class TorchVisionDataset(torch.utils.data.Dataset):
    def __init__(self, input):
        self.input = input

    def __getitem__(self, idx):
        item = {'x': self.input[i][0].clone().detach()}
        return (item, self.input[i][1].clone().detach() if self.input[i][1] != None else None)

    def __len__(self):
        return len(self.input)

class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings['input_ids'][idx].clone().detach().to(device),
            "attention_mask": self.encodings['attention_mask'][idx].clone().detach().to(device),
        }
        return (item, torch.tensor([self.labels[idx].clone().detach()]).to(device) if self.labels != None else torch.tensor([0]*len(item['input_ids'])).to(device))

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
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

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