# BucketSampler from https://github.com/PetrochukM/PyTorch-NLP

import heapq
import math
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import RandomSampler
import random
import torch
import collections
import inspect
from torch import nn
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import json

def set_seed(seed=6750):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def load_model(model: nn.Module, path: Path, multi2single=False) -> Tuple:
    state = torch.load(str(path))
    if multi2single:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['model'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
    return state, state['best_valid_loss']
    
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
       param_group['lr'] = lr

    return True



def _biggest_batches_first(o):
    return sum([t.numel() for t in get_tensors(o)])


def _identity(e):
    return e

from torch.utils.data.sampler import Sampler


def _identity(e):
    return e

def get_tensors(object_):
    """ Get all tensors associated with ``object_``
    Args:
        object_ (any): Any object to look for tensors.
    Returns:
        (list of torch.tensor): List of tensors that are associated with ``object_``.
    """
    if torch.is_tensor(object_):
        return [object_]
    elif isinstance(object_, (str, float, int)):
        return []

    tensors = set()

    if isinstance(object_, collections.abc.Mapping):
        for value in object_.values():
            tensors.update(get_tensors(value))
    elif isinstance(object_, collections.abc.Iterable):
        for value in object_:
            tensors.update(get_tensors(value))
    else:
        members = [
            value for key, value in inspect.getmembers(object_)
            if not isinstance(value, (collections.abc.Callable, type(None)))
        ]
        tensors.update(get_tensors(members))

    return tensors

class ShuffleBatchSampler(BatchSampler):
    """Wraps another sampler to yield a mini-batch of indices.
    The ``ShuffleBatchSampler`` adds ``shuffle`` on top of
    ``torch.utils.data.sampler.BatchSampler``.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        shuffle (bool, optional): If ``True``, the sampler will shuffle the batches.
    Example:
        >>> import random
        >>> from torchnlp.samplers import SortedSampler
        >>>
        >>> random.seed(123)
        >>>
        >>> list(ShuffleBatchSampler(SortedSampler(range(10)), batch_size=3, drop_last=False))
        [[6, 7, 8], [9], [3, 4, 5], [0, 1, 2]]
        >>> list(ShuffleBatchSampler(SortedSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [6, 7, 8], [3, 4, 5]]
    """

    def __init__(
            self,
            sampler,
            batch_size,
            drop_last,
            shuffle=True,
    ):
        self.shuffle = shuffle
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        # NOTE: This is not data
        batches = list(super().__iter__())
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

class SortedSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self, data, sort_key=_identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)

class BucketBatchSampler(object):
    """ Batches are sampled from sorted buckets of data.

    We use a bucketing technique from ``torchtext``. First, partition data in buckets of size
    100 * ``batch_size``. The examples inside the buckets are sorted using ``sort_key`` and batched.

    **Background**

        BucketBatchSampler is similar to a BucketIterator found in popular libraries like `AllenNLP`
        and `torchtext`. A BucketIterator pools together examples with a similar size length to
        reduce the padding required for each batch. BucketIterator also includes the ability to add
        noise to the pooling.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        data (iterable): Data to sample from.
        batch_size (int): Size of mini-batch.
        sort_key (callable): specifies a function of one argument that is used to extract a
          comparison key from each list element
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        biggest_batch_first (callable or None, optional): If a callable is provided, the sampler
            approximates the memory footprint of tensors in each batch, returning the 5 biggest
            batches first. Callable must return a number, given an example.

            This is largely for testing, to see how large of a batch you can safely use with your
            GPU. This will let you try out the biggest batch that you have in the data `first`, so
            that if you're going to run out of memory, you know it early, instead of waiting
            through the whole epoch to find out at the end that you're going to crash.

            Credits:
            https://github.com/allenai/allennlp/blob/3d100d31cc8d87efcf95c0b8d162bfce55c64926/allennlp/data/iterators/bucket_iterator.py#L43
        bucket_size_multiplier (int): Batch size multiplier to determine the bucket size.
        shuffle (bool, optional): If ``True``, the sampler will shuffle the batches.

    Example:
        >>> list(BucketBatchSampler(range(10), batch_size=3, drop_last=False))
        [[9], [3, 4, 5], [6, 7, 8], [0, 1, 2]]
        >>> list(BucketBatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    """

    def __init__(
            self,
            data,
            batch_size,
            drop_last,
            sort_key=_identity,
            biggest_batches_first=_biggest_batches_first,
            bucket_size_multiplier=100,
            shuffle=True,
    ):
        self.biggest_batches_first = biggest_batches_first
        self.sort_key = sort_key
        self.bucket_size_multiplier = bucket_size_multiplier
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data = data
        self.shuffle = shuffle

        self.bucket_size_multiplier = bucket_size_multiplier
        self.bucket_sampler = BatchSampler(
            RandomSampler(data), batch_size * bucket_size_multiplier, False)

    def __iter__(self):

        def get_batches():
            """ Get bucketed batches """
            for bucket in self.bucket_sampler:
                for batch in ShuffleBatchSampler(
                        SortedSampler(bucket, lambda i: self.sort_key(self.data[i])),
                        self.batch_size,
                        drop_last=self.drop_last,
                        shuffle=self.shuffle):
                    batch = [bucket[i] for i in batch]

                    # Should only be triggered once
                    if len(batch) < self.batch_size and self.drop_last:
                        continue

                    yield batch

        if self.biggest_batches_first is None:
            return get_batches()
        else:
            batches = list(get_batches())
            biggest_batches = heapq.nlargest(
                5,
                range(len(batches)),
                key=lambda i: sum([self.biggest_batches_first(self.data[j]) for j in batches[i]]))
            front = [batches[i] for i in biggest_batches]
            # Remove ``biggest_batches`` from data
            for i in sorted(biggest_batches, reverse=True):
                batches.pop(i)
            # Move them to the front
            batches[0:0] = front
            return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.data) // self.batch_size
        else:
            return math.ceil(len(self.data) / self.batch_size)

def write_event(log, step: int, epoch=None, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    data['epoch']=epoch
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()