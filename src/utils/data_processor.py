import csv
import os
import sys
import pandas as pd
import numpy as np

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
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


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


na_list = ["*", "*?", "??", "?*"]

class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b, label=label)
            )
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(
                data_dir, "dev_matched.tsv")), "dev_matched"
        )

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b, label=label)
            )
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(
                data_dir, "dev_mismatched.tsv")), "dev_matched"
        )


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_train_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', index_col=False, names=["misc", "label", "misc2", "sentence"])[["label", "sentence"]]
        # print(tsv)
        # exit()
        tsv = tsv.dropna()
        # tsv['label'] = tsv['label'].astype(float)
        return tsv

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_dev_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', index_col=False, names=["misc", "label", "misc2", "sentence"])[["label", "sentence"]]
        tsv = tsv.dropna()
        # tsv['label'] = tsv['label'].astype(float)
        return tsv

    def get_test_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv = tsv.rename(columns={"index": "id"})
        return tsv

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=None, label=label)
            )
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_train_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', index_col=False, header=0)[["label", "sentence"]]
        tsv = tsv.dropna()
        # tsv['label'] = tsv['label'].astype(float)
        return tsv

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_dev_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', index_col=False, header=0)[["label", "sentence"]]
        tsv = tsv.dropna()
        # tsv['label'] = tsv['label'].astype(float)
        return tsv

    def get_test_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv = tsv.rename(columns={"index": "id"})
        return tsv

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=None, label=label)
            )
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b, label=label)
            )
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_train_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv['sentence'] = tsv.question1 + " [SEP] " + tsv.question2
        tsv = tsv.rename(columns={"is_duplicate": "label"})
        tsv = tsv[["label", "sentence"]]
        return tsv

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_dev_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv['sentence'] = tsv.question1 + " [SEP] " + tsv.question2
        tsv = tsv.rename(columns={"is_duplicate": "label"})
        tsv = tsv[["label", "sentence"]]
        return tsv

    def get_test_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv['sentence'] = tsv.question1 + " [SEP] " + tsv.question2
        # tsv = tsv.rename(columns={"is_duplicate": "label"})
        tsv = tsv[["id", "sentence"]]
        return tsv

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b, label=label)
            )
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_train_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', index_col=False, header=0, encoding='utf-8', error_bad_lines=False)
        tsv = tsv.dropna()
        tsv['num_label'] = np.where(tsv.label == 'not_entailment', 1, 0)
        tsv['sentence'] = tsv.question + " [SEP] " + tsv.sentence
        tsv = tsv[["num_label", "sentence"]]
        tsv = tsv.rename(columns={'num_label': 'label'})
        return tsv

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched"
        )

    def get_dev_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', index_col=False, header=0, encoding='utf-8', error_bad_lines=False)
        tsv = tsv.dropna()
        tsv['num_label'] = np.where(tsv.label == 'not_entailment', 1, 0)
        tsv['sentence'] = " [CLS] " + tsv.question + " [SEP] " + tsv.sentence + " [SEP] "
        tsv = tsv[["num_label", "sentence"]]
        tsv = tsv.rename(columns={'num_label': 'label'})
        return tsv

    def get_test_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', index_col=False, header=0, encoding='utf-8', error_bad_lines=False)
        tsv = tsv.dropna()
        tsv['sentence'] = " [CLS] " + tsv.question + " [SEP] " + tsv.sentence + " [SEP] "
        tsv = tsv[["index", "sentence"]]
        tsv = tsv.rename(columns={'index': 'id'})
        return tsv

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b, label=label)
            )
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_train_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv['num_label'] = np.where(tsv.label == 'not_entailment', 1, 0)
        tsv['sentence'] = tsv.sentence1 + " [SEP] " + tsv.sentence2
        tsv = tsv[["num_label", "sentence"]]
        tsv = tsv.rename(columns={'num_label': 'label'})
        return tsv

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_dev_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv['num_label'] = np.where(tsv.label == 'not_entailment', 1, 0)
        tsv['sentence'] = tsv.sentence1 + " [SEP] " + tsv.sentence2
        tsv = tsv[["num_label", "sentence"]]
        tsv = tsv.rename(columns={'num_label': 'label'})
        return tsv

    def get_test_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t', index_col=False, header=0, error_bad_lines=False)
        tsv = tsv.dropna()
        tsv['sentence'] = " [CLS] " + tsv.sentence1 + " [SEP] " + tsv.sentence2 + " [SEP] "
        tsv = tsv[["index", "sentence"]]
        tsv = tsv.rename(columns={'index': 'id'})
        return tsv

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b, label=label)
            )
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_train_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv['sentence'] = tsv.sentence1 + " [SEP] " + tsv.sentence2
        tsv = tsv[["label", "sentence"]]
        return tsv

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_dev_tsv(self, data_dir):
        tsv = pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t', index_col=False, header=0)
        tsv = tsv.dropna()
        tsv['sentence'] = tsv.sentence1 + " [SEP] " + tsv.sentence2
        tsv = tsv[["label", "sentence"]]
        return tsv

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a,
                             text_b=text_b, label=label)
            )
        return examples


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

bert_base_model_config = {
    "vocab_size_or_config_json_file": 119547,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 16,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "initializer_range": 0.02,
}
