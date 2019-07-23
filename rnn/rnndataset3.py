import argparse
import random

import pandas as pd
import tqdm
import numpy as np
from pathlib import Path
from pytorch_pretrained_bert import BertTokenizer

from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
from sklearn.feature_extraction.text import CountVectorizer

VOCAB_PATH = Path('../input/preparedRNNData/word_dict.pkl')
VECTORIZER_PATH = Path('../input/preparedRNNData/cntv.pkl')
def convert_one_line(text, max_seq_length=None, word_dict=None, random_unknown=0.05):
    if len(text) > max_seq_length:
        # tokens_a = tokens_a[:max_seq_length]
        text = text[:max_seq_length // 2] + text[-(max_seq_length - max_seq_length // 2):]
    one_token = []
    for t in text:
        if np.random.rand()<random_unknown:
            one_token.append(1)
        else:
            if t in word_dict:
                one_token.append(word_dict[t])
            else:
                one_token.append(1) # 1 for unknown
    return one_token

identity_words = [
        set(['gay','homosexual','lesbian','lgbt','bisexual','heterosexual','bisexual','homosexuals']),
        set(['muslim','islam','islamic','muslims']),
        set(['jewish','christian','palestinian','jew','jews','church','christians','christianity','catholics','catholic']),
        set(['psychiatric','mental'])
    ]

def get_num_features(text):
    retval = np.zeros(12)
    retval[0] = len(text)
    retval[1] = len(set(text))
    retval[2] = len(set(text))/retval[0]
    for w in text:
        retval[11]+=len(w)
        temp = w.lower()
        for idx, word_set in enumerate(identity_words):
            if temp in word_set:
                retval[idx + 3]+=1
                retval[idx + 7] += 1/retval[0]
    return retval

class RNNDataset(Dataset):

    def __init__(self, text, features, target, vocab_path = VOCAB_PATH,
                 random_unknown=0.05):
        super(RNNDataset, self).__init__()

        self._text = text
        self._features = features
        self._target = target
        self.random_unknown=random_unknown
        with open(vocab_path,'rb') as f:
            self._word_dict = pickle.load(f)

    def __len__(self):
        return len(self._text)

    def __getitem__(self, idx):
        text = self._text[idx]
        target = self._target[idx]
        feat = self._features[idx]
        text = convert_one_line(text, max_seq_length=300, word_dict=self._word_dict,
                                                 random_unknown=self.random_unknown)

        return torch.LongTensor(text), feat, len(text), target

def collate_fn(batch):
    text, feats, lens, targets = zip(*batch)
    text = pad_sequence(text, batch_first=True)
    feats = torch.FloatTensor(feats)
    lens = torch.LongTensor(lens)
    targets = torch.FloatTensor(targets)#.view(-1,1)
    return text, feats, lens, targets
