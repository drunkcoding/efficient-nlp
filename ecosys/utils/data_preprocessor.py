
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, SequentialSampler
import torch

from .data_structure import HuggingFaceDataset

def build_gpt2_special_tokens(self, token_ids_0, token_ids_1=None):
    """
    For GPT2 special must be added after text tokenization
    """
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    if token_ids_1 != None:
        outputs += token_ids_1 + [self.eos_token_id]
    return outputs 

class GPT2TokenizerPadded(GPT2Tokenizer):
    """
    Additional padding token added to GPT2 tokenization
    """
    def __init__(self, vocab_file, merges_file, errors="replace", unk_token="<|endoftext|>", bos_token="<|endoftext|>", eos_token="<|endoftext|>", add_prefix_space=False, **kwargs):
        super().__init__(vocab_file, merges_file, errors=errors, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, add_prefix_space=add_prefix_space, **kwargs)

        self.build_inputs_with_special_tokens = build_gpt2_special_tokens
        self.pad_token = self.unk_token


def build_image_dataloader():
    pass


def build_texts_dataloader(tokenizer, texts, labels=None, batch_size=32, **kwargs):
    encoded_texts = {
        "input_ids": torch.Tensor(list()),
        "attension_mask": torch.Tensor(list()),
    }
    
    for text in texts:
        token_ids = tokenizer(
            text, 
            **kwargs
        )
        for token_id in token_ids:

    dataset = HuggingFaceDataset(encoded_texts, labels)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size
    )
    return dataloader