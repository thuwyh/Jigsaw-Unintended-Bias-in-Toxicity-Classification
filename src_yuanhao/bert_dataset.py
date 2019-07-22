import argparse
import random

import pandas as pd
import tqdm

from pathlib import Path
from pytorch_pretrained_bert import BertTokenizer

from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

VOCAB_PATH = Path('../input/torch-bert-weights/bert-base-uncased-vocab.txt')

def convert_one_line(text, max_seq_length=None, tokenizer=None):
    max_seq_length -= 2
    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > max_seq_length:
        # tokens_a = tokens_a[:max_seq_length]
        tokens_a = tokens_a[:max_seq_length // 2] + tokens_a[-(max_seq_length - max_seq_length // 2):]
    one_token = tokenizer.convert_tokens_to_ids(
        ["[CLS]"]+tokens_a+["[SEP]"])#+[0] * (max_seq_length - len(tokens_a))
    return one_token


class TrainDataset(Dataset):

    def __init__(self, text, lens, weights, target, vocab_path = VOCAB_PATH, do_lower = True):
        super(TrainDataset, self).__init__()

        self._text = text
        self._lens = lens
        self._target = target
        self._weights = weights
        self._tokenizer = BertTokenizer.from_pretrained(
            vocab_path, cache_dir=None, do_lower_case=do_lower)

    def __len__(self):
        return len(self._text)

    def __getitem__(self, idx):
        text = self._text[idx]
        lens = self._lens[idx]
        target = self._target[idx]
        weight = self._weights[idx]

        return torch.LongTensor(convert_one_line(text, max_seq_length=220, tokenizer=self._tokenizer)), lens, weight, target

def collate_fn(batch):
    text, lens, weights, targets = zip(*batch)
    text = pad_sequence(text, batch_first=True)
    lens = torch.LongTensor(lens) 
    weights = torch.FloatTensor(weights)
    targets = torch.FloatTensor(targets)#.view(-1,1)
    return text, lens, weights, targets
