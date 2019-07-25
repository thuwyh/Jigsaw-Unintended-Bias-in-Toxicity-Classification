import argparse
import json
import os
import shutil
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import tqdm
from apex import amp
from pytorch_pretrained_bert import BertAdam, BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import (
    BucketBatchSampler,
    get_learning_rate, set_seed,
    write_event, load_model)

# loss1
def custom_loss(pred, targets, loss_weight):
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(pred[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(pred[:, 1:], targets[:, 2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2


# loss2
def custom_loss2(pred, targets, loss_weight):
    p_mean = 0.6568
    n_mean = 0.0549
    p_squ = 0.45082
    n_squ = 0.01456

    pred_ = pred[:, :1]
    P = targets[:, :1] != 0
    N = targets[:, :1] == 0

    x_p, x_n = pred_[P], pred_[N]
    if not x_p.size()[0]:
        mean_p = p_mean
        squ_p = p_squ
    else:
        mean_p = x_p.mean()
        squ_p = (x_p * x_p).mean()

    if not x_n.size():
        mean_n = n_mean
        squ_n = n_squ
    else:
        mean_n = x_n.mean()
        squ_n = (x_n * x_n).mean()

    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(pred[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(pred[:, 1:], targets[:, 2:])
    # extra_loss_1 = torch.log(torch.exp(mean_n - mean_p) + 1)
    extra_loss_1 = mean_n - mean_p + 1

    return (bce_loss_1 * loss_weight) + bce_loss_2 + extra_loss_1 * 0.1


def convert_one_line(text, max_seq_length=None, tokenizer=None, split_point=0.25):
    max_seq_length -= 2
    tokens_a = tokenizer.tokenize(text)
    int_split = int(split_point * max_seq_length)
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[:int_split] + tokens_a[int_split - max_seq_length:]
    one_token = tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + tokens_a + ["[SEP]"])  # +[0] * (max_seq_length - len(tokens_a))
    return one_token


class TrainDataset(Dataset):

    def __init__(self, text, lens, target, identity_df, weights, model="mybert", split_point=0.25):
        super(TrainDataset, self).__init__()

        self._text = text
        self._lens = lens
        self._target = target
        self._identity_df = identity_df
        self._weights = weights
        self._split_point = split_point
        VOCAB_PATH = Path('../input/torch-bert-weights/%s-vocab.txt' % (model))

        if model in ["bert-base-uncased", "bert-large-uncased", "mybert", "mybert-large-uncased", "mybert-wwm-uncased",
                     'mybert-base-uncased']:
            do_lower_case = True
        elif model in ["bert-base-cased", "bert-large-cased", "mybertlargecased", "mybert-base-cased"]:
            do_lower_case = False
        else:
            raise ValueError('%s is not a valid model' % model)

        self._tokenizer = BertTokenizer.from_pretrained(
            VOCAB_PATH, cache_dir=None, do_lower_case=do_lower_case)

    def __len__(self):
        return len(self._text)

    def __getitem__(self, idx):
        text = self._text[idx]
        lens = self._lens[idx]
        target = self._target[idx]
        # identity_df = self._identity_df.iloc[[idx], :]
        weight = self._weights[idx]
        return torch.LongTensor(convert_one_line(text, max_seq_length=220, tokenizer=self._tokenizer,
                                                 split_point=self._split_point)), lens, target, weight


def collate_fn(batch):
    text, lens, targets, weights = zip(*batch)
    # identity_df = pd.concat(list(identitys)).reset_index(drop=True)
    text = pad_sequence(text, batch_first=True)
    lens = torch.LongTensor(lens)
    weights = torch.FloatTensor(weights)
    targets = torch.FloatTensor(targets)
    return text, lens, targets, weights


class BertModel(nn.Module):

    def __init__(self, pretrain_path, dropout=0.1):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(pretrain_path, cache_dir=None, num_labels=1)
        self.aux_head = nn.Sequential(
            OrderedDict([
                ('dropout', nn.Dropout(dropout)),
                ('clf', nn.Linear(self.bert.config.hidden_size, 6)),
            ])
        )
        self.main_head = nn.Sequential(
            OrderedDict([
                ('dropout', nn.Dropout(dropout)),
                ('clf', nn.Linear(self.bert.config.hidden_size, 1))
            ])
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        aux_logits = self.aux_head(pooled_output)
        main_logits = self.main_head(pooled_output)
        out = torch.cat([main_logits, aux_logits], 1)
        return out


class JigsawEvaluator:

    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25, identity_columns=None):
        self.y = (y_true >= 0.5).astype(int)
        self.y_i = (y_identity >= 0.5).astype(int)
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight
        self.identity_columns = identity_columns

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = []
        # records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            record = dict()
            record['subgroup_auc'] = self._compute_subgroup_auc(i, y_pred)
            record['bpsn_auc'] = self._compute_bpsn_auc(i, y_pred)
            record['bnsp_auc'] = self._compute_bnsp_auc(i, y_pred)
            # records[0, i] = self._compute_subgroup_auc(i, y_pred)
            # records[1, i] = self._compute_bpsn_auc(i, y_pred)
            # records[2, i] = self._compute_bnsp_auc(i, y_pred)
            records.append(record)
        return pd.DataFrame(records, index=self.identity_columns)

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            # self._power_mean(bias_metrics[0]),
            # self._power_mean(bias_metrics[1]),
            # self._power_mean(bias_metrics[2])
            self._power_mean(bias_metrics['subgroup_auc']),
            self._power_mean(bias_metrics['bpsn_auc']),
            self._power_mean(bias_metrics['bnsp_auc'])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score, bias_metrics


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict', 'train_all'])
    arg('run_root')
    arg('--model', default='mybert')
    arg('--pretrained', type=int, default=0)
    arg('--batch-size', type=int, default=32)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2)
    arg('--lr', type=float, default=0.0002)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=1)
    arg('--kloss', type=float, default=1.0)
    arg('--loss_fn', default='loss1')
    arg('--fold_name', default='/folds_binary_weights_kernal.pkl')
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--multi-gpu', type=int, default=0)
    arg('--lr_layerdecay', type=float, default=0.95)
    arg('--warmup', type=float, default=0.05)
    arg('--split_point', type=float, default=0.3)
    arg('--bsample', type=bool, default=True)
    args = parser.parse_args()

    set_seed()
    BERT_PRETRAIN_PATH = '../input/torch-bert-weights/%s/' % (args.model)
    run_root = Path('../experiments/' + args.run_root)
    DATA_ROOT = Path('../input/jigsaw-unintended-bias-in-toxicity-classification')

    folds = pd.read_pickle(DATA_ROOT / 'folds.pkl')

    identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    weights = np.ones((len(folds),)) / 4
    # # Subgroup
    weights += (folds[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # # Background Positive, Subgroup Negative
    weights += (((folds['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (folds[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    # # Background Negative, Subgroup Positive
    weights += (((folds['target'].values < 0.5).astype(bool).astype(np.int) +
                 (folds[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4

    folds['weights'] = weights
    print(folds['weights'].mean())

    if args.mode == "train_all":
        train_fold = folds
    else:
        train_fold = folds[folds['fold'] != args.fold]
        valid_fold = folds[folds['fold'] == args.fold]
        valid_fold = valid_fold.sort_values(by=["len"])

    if args.limit:
        train_fold = train_fold[:args.limit]
        if args.mode != "train_all":
            valid_fold = valid_fold[:args.limit * 3]

    if args.mode == "train_all":
        valid_df = None
    else:
        valid_df = valid_fold[identity_columns + ["target"]]

    loss_weight = 1 / folds['weights'].mean() * args.kloss

    if args.loss_fn == "loss1":
        loss_fn = custom_loss
    elif args.loss_fn == "loss2":
        loss_fn = custom_loss2

    criterion = partial(loss_fn, loss_weight=loss_weight)

    if args.mode == 'train' or args.mode == "train_all":
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        training_set = TrainDataset(train_fold['comment_text'].tolist(), lens=train_fold['len'].tolist(),
                                    target=train_fold[['binary_target', 'weights', 'target', 'severe_toxicity',
                                                       'obscene', 'identity_attack', 'insult',
                                                       'threat']].values.tolist(),
                                    identity_df=train_fold[identity_columns], weights=train_fold['weights'].tolist(),
                                    model=args.model, split_point=args.split_point)
        if args.bsample:
            bbsampler = BucketBatchSampler(training_set, batch_size=args.batch_size, drop_last=True,
                                           sort_key=lambda x: x[1], biggest_batches_first=None,
                                           bucket_size_multiplier=100, shuffle=True)
            batchsize = 1
            shuffle = False

        else:
            bbsampler = None
            batchsize = args.batch_size
            shuffle = True

        training_loader = DataLoader(training_set, batch_sampler=bbsampler, collate_fn=collate_fn,
                                     num_workers=args.workers, batch_size=batchsize, shuffle=shuffle)

        if args.mode == "train":
            valid_set = TrainDataset(valid_fold['comment_text'].tolist(), lens=valid_fold['len'].tolist(),
                                     target=valid_fold['binary_target'].values.tolist()
                                     , identity_df=valid_fold[identity_columns], weights=valid_fold['weights'].tolist(),
                                     model=args.model, split_point=args.split_point)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                      num_workers=args.workers)
        else:
            valid_loader = None

        # model = BertForSequenceClassification.from_pretrained(BERT_PRETRAIN_PATH,cache_dir=None,num_labels=1)
        model = BertModel(BERT_PRETRAIN_PATH)
        model.cuda()

        if args.model in ["bert-base-uncased", "bert-base-cased", "mybert", "gpt2", 'mybert-base-cased',
                          'mybert-base-uncased']:
            NUM_LAYERS = 12
        elif args.model in ["bert-large-uncased", "bert-large-cased", "mybertlarge", "wmm", "mybertlargecased",
                            "mybert-large-uncased", 'mybert-wwm-uncased']:
            NUM_LAYERS = 24
        else:
            raise ValueError('%s is not a valid model' % args.model)

        optimizer_grouped_parameters = [
            {'params': model.bert.bert.embeddings.parameters(), 'lr': args.lr * (args.lr_layerdecay ** NUM_LAYERS)},
            {'params': model.main_head.parameters(), 'lr': args.lr},
            {'params': model.aux_head.parameters(), 'lr': args.lr},
            {'params': model.bert.bert.pooler.parameters(), 'lr': args.lr}
        ]

        for layer in range(NUM_LAYERS):
            optimizer_grouped_parameters.append(
                {'params': model.bert.bert.encoder.layer.__getattr__('%d' % (NUM_LAYERS - 1 - layer)).parameters(),
                 'lr': args.lr * (args.lr_layerdecay ** layer)},
            )
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup,
                             t_total=len(training_loader) // args.step)

        scheduler = ReduceLROnPlateau(optimizer, patience=0, factor=0.1, verbose=True, mode='max', min_lr=1e-7)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", verbosity=0)

        optimizer.zero_grad()

        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        train(args, model, optimizer, scheduler, criterion,
              train_loader=training_loader,
              valid_df=valid_df, valid_loader=valid_loader, epoch_length=len(training_set))

    elif args.mode == 'validate':

        valid_set = TrainDataset(valid_fold['comment_text'].tolist(), lens=valid_fold['len'].tolist(),
                                 target=valid_fold[['binary_target']].values.tolist(),
                                 identity_df=valid_fold[identity_columns],
                                 weights=valid_fold['weights'].tolist(), model=args.model, split_point=args.split_point)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  num_workers=args.workers)
        model = BertModel(BERT_PRETRAIN_PATH)
        load_model(model, run_root / ('best-model-%d.pt' % args.fold), multi2single=False)
        model.cuda()

        optimizer = BertAdam(model.parameters(), lr=1e-5, warmup=0.95)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", verbosity=0)

        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        validation(model, criterion, valid_df, valid_loader, args, save_result=True, progress=True)


def train(args, model: nn.Module, optimizer, scheduler, criterion, *,
          train_loader, valid_df, valid_loader, epoch_length, patience=1,
          n_epochs=None) -> bool:
    n_epochs = n_epochs or args.n_epochs

    run_root = Path('../experiments/' + args.run_root)
    model_path = run_root / ('model-%d.pt' % args.fold)
    best_model_path = run_root / ('best-model-%d.pt' % args.fold)
    if best_model_path.exists():
        state, best_valid_score = load_model(model, best_model_path)
        start_epoch = state['epoch']
        best_epoch = start_epoch
    else:
        best_valid_score = 0
        start_epoch = 0
        best_epoch = 0
    step = 0

    if args.mode == "train_all":
        current_score = 0.95

    save = lambda ep: torch.save({
        'model': model.module.state_dict() if args.multi_gpu == 1 else model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': current_score
    }, str(model_path))
    #
    report_each = 10000
    log = run_root.joinpath('train-%d.log' % args.fold).open('at', encoding='utf8')

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()

        lr = get_learning_rate(optimizer)
        tq = tqdm.tqdm(total=epoch_length)
        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []

        mean_loss = 0
        for i, (inputs, _, targets, weights) in enumerate(train_loader):
            attention_mask = (inputs > 0).cuda()
            inputs, targets, weights = inputs.cuda(), targets.cuda(), weights.unsqueeze(1).cuda()

            outputs = model(inputs, attention_mask=attention_mask, labels=None)

            loss = criterion(outputs, targets) / args.step
            # loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, targets)
            batch_size = inputs.size(0)
            # loss.backward()
            if (i + 1) % args.step == 0:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                with amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

            tq.update(batch_size)
            losses.append(loss.item() * args.step)
            mean_loss = np.mean(losses[-report_each:])
            tq.set_postfix(loss=f'{mean_loss:.5f}')
            if i and i % report_each == 0:
                write_event(log, step, loss=mean_loss)

        write_event(log, step, epoch=epoch, loss=mean_loss)
        tq.close()

        # if epoch<7: continue
        if args.mode == "train":
            valid_metrics = validation(model, criterion, valid_df, valid_loader, args)
            write_event(log, step, **valid_metrics)
            current_score = valid_metrics['score']
        save(epoch + 1)
        if scheduler is not None and args.mode == "train":
            scheduler.step(current_score)

        if args.mode == "train":
            if current_score > best_valid_score:
                best_valid_score = current_score
                shutil.copy(str(model_path), str(best_model_path))
                best_epoch = epoch
            else:
                pass
        # if isinstance(criterion,nn.BCEWithLogitsLoss) and lr<0.00002:
        # break
    return True


def validation(model: nn.Module, criterion, valid_df, valid_loader, args, save_result=True, progress=True) -> Dict[
    str, float]:
    run_root = Path('../experiments/' + args.run_root)
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for inputs, _, targets, weights in valid_loader:
            if progress:
                batch_size = inputs.size(0)
                tq.update(batch_size)
            all_targets.append(targets.numpy().copy())
            attention_mask = (inputs > 0).cuda()
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, attention_mask=attention_mask, labels=None)
            outputs = outputs[:, 0].view(-1, 1)
            loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))  # *N_CLASSES
            all_losses.append(loss.item())  # _reduce_loss
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    if save_result:
        np.save(run_root / 'prediction_fold{}.npy'.format(args.fold), all_predictions)
        np.save(run_root / 'target_fold{}.npy'.format(args.fold), all_targets)

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    y_true = valid_df['target'].values
    y_identity = valid_df[identity_columns].values

    evaluator = JigsawEvaluator(y_true, y_identity, identity_columns=identity_columns)

    score, bias_metrics = evaluator.get_final_metric(all_predictions)
    if progress:
        tq.close()
    # valid_copy = valid_df.copy()
    # valid_copy['model'] = all_predictions
    # bias_metrics_df = compute_bias_metrics_for_model(valid_copy, 'model', label_col='target')
    # score = get_final_metric(bias_metrics_df, calculate_overall_auc(valid_copy, 'model'))

    metrics = dict()
    metrics['loss'] = np.mean(all_losses)
    metrics['score'] = score
    to_print = []
    for idx, (k, v) in enumerate(sorted(metrics.items(), key=lambda kv: -kv[1])):
        to_print.append(f'{k} {v:.5f}')
    print(' | '.join(to_print))
    metrics["bias"] = bias_metrics.to_json()
    return metrics


if __name__ == '__main__':
    main()
