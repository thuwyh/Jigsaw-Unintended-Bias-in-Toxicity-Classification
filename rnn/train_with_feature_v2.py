import argparse
import json
import shutil
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tqdm
import os
import gc

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR
from torch.utils.data import DataLoader
from apex import amp

from utils import (
    BucketBatchSampler,
    get_learning_rate, set_learning_rate, set_seed,
    write_event, load_model)

from metric2 import JigsawEvaluator, custom_loss

from rnndataset3 import RNNDataset, collate_fn, get_num_features
from rnnmodelv3 import NeuralNetV2

import torch

ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ

def custom_loss_mtl(pred, targets):
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1])(pred[:, 0], targets[:, 0])  #
    bce_loss_2 = nn.BCEWithLogitsLoss()(pred[:, 1], targets[:, 2]) * 20
    bce_loss_4 = nn.BCEWithLogitsLoss()(pred[:, 3:10], targets[:, 2:9]) * 20

    return bce_loss_1+bce_loss_2+bce_loss_4   # +bce_loss_5+mse_loss_6 +bce_loss_2


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict'])
    arg('run_root')
    arg('--batch-size', type=int, default=32)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2)
    arg('--lr', type=float, default=0.0002)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=8)
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--multi-gpu', type=int, default=0)
    args = parser.parse_args()

    set_seed()

    run_root = Path('../experiments/' + args.run_root)
    DATA_ROOT = Path('../input/jigsaw-unintended-bias-in-toxicity-classification')

    folds = pd.read_pickle(DATA_ROOT / 'folds.pkl')

    ## weights
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

    train_fold = folds[folds['fold'] != args.fold]
    valid_fold = folds[folds['fold'] == args.fold]
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    data_dir = Path('../input/preparedRNNData')
    with open(data_dir / 'training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)

    training_text = [training_data[i] for i in train_fold.index.tolist()]
    validation_text = [training_data[i] for i in valid_fold.index.tolist()]

    VECTORIZER_PATH = Path('../input/preparedRNNData/cntv.pkl')
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

    features = np.load(data_dir/'features.npy')
    valid_features = features[valid_fold.index.tolist(),:]

    valid_df = valid_fold[identity_columns + ['target']]
    criterion = custom_loss_mtl

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        with open(data_dir / 'glove_embedding.npy', 'rb') as f:
            glove_embedding = np.load(f)

        with open(data_dir / 'crawl_embedding.npy', 'rb') as f:
            crawl_embedding = np.load(f)

        with open(data_dir / 'my_embedding.npy', 'rb') as f:
            my_embedding = np.load(f)

        init_embedding = np.concatenate([glove_embedding, crawl_embedding, my_embedding], axis=-1)
        print('embedding size:', init_embedding.shape)
        del glove_embedding
        del crawl_embedding
        del my_embedding
        gc.collect()

        train_features = features[train_fold.index.tolist(),:]

        training_set = RNNDataset(training_text, features=train_features,
                                    target=train_fold[['binary_target', 'weights', 'target', 'severe_toxicity',
                                                       'obscene', 'identity_attack', 'insult', 'threat',
                                                       'sexual_explicit']+identity_columns].values.tolist(),
                                  random_unknown=0.02)
        training_loader = DataLoader(training_set, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers)  #batch_sampler=bbsampler,

        valid_set = RNNDataset(validation_text, features=valid_features,
                                 target=valid_fold['binary_target'].tolist(),
                               random_unknown=-1)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  num_workers=args.workers)

        model = NeuralNetV2(init_embedding=init_embedding, max_features = init_embedding.shape[0],
                          embed_size=init_embedding.shape[1], hidden_size=128)
        model.cuda()

        optimizer = Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr) #)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=0, factor=0.2,mode='max')
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        optimizer.zero_grad()

        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        train(args, model, optimizer, scheduler, criterion,
              train_loader=training_loader,
              valid_df=valid_df, valid_loader=valid_loader, epoch_length=len(training_set))

    elif args.mode == 'validate':
        valid_set = RNNDataset(validation_text, features=valid_features,
                                target=valid_fold['binary_target'].tolist(),
                               random_unknown=-1)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  num_workers=args.workers)
        model = NeuralNetV2(init_embedding=None, max_features=518136, embed_size=700,
                          hidden_size=128)
        load_model(model, run_root / ('best-model-%d.pt' % args.fold), multi2single=False)
        model.cuda()
        if args.multi_gpu == 1:
            model = nn.DataParallel(model)
        validation(model, criterion, valid_df, valid_loader, args, True, progress=True)


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
        for i, (inputs, feats, lens, targets) in enumerate(train_loader):
            inputs, feats, lens, targets = inputs.cuda(), feats.cuda(), lens.cuda(), targets.cuda()

            outputs = model(inputs, feats, lens)
            loss = criterion(outputs, targets)/args.step
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

        valid_metrics = validation(model, criterion, valid_df, valid_loader, args)
        write_event(log, step, **valid_metrics)
        current_score = valid_metrics['score']
        save(epoch + 1)
        if scheduler is not None:
            scheduler.step(current_score)#current_score
        if current_score > best_valid_score:
            best_valid_score = current_score
            shutil.copy(str(model_path), str(best_model_path))
            best_epoch = epoch
        else:
            pass
        if isinstance(criterion, nn.BCEWithLogitsLoss) and lr < 0.00002:
            break
    return True


def validation(model: nn.Module, criterion, valid_df, valid_loader, args, save_result=False, progress=False) -> Dict[
    str, float]:
    run_root = Path('../experiments/' + args.run_root)
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []

    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for inputs, feats, lens, targets in valid_loader:
            if progress:
                batch_size = inputs.size(0)
                tq.update(batch_size)
            all_targets.append(targets.numpy().copy())
            inputs, feats, lens, targets = inputs.cuda(), feats.cuda(), lens.cuda(), targets.cuda()
            outputs = model(inputs, feats, lens)

            predictions = torch.sigmoid(outputs[:, 0].view(-1, 1))
            all_predictions.append(predictions.cpu().numpy())

            outputs = outputs[:, 0].view(-1, 1)
            loss = nn.BCEWithLogitsLoss()(outputs,
                                          targets.view(-1, 1))  # criterion(outputs, targets).mean()  # *N_CLASSES
            all_losses.append(loss.item())  # _reduce_loss
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

    print('bce')
    score, bias_metrics = evaluator.get_final_metric(all_predictions)
    print(bias_metrics)
    print(score)

    metrics = dict()
    metrics['loss'] = np.mean(all_losses)
    metrics['score'] = score
    to_print = []
    for idx, (k, v) in enumerate(sorted(metrics.items(), key=lambda kv: -kv[1])):
        to_print.append(f'{k} {v:.5f}')
    print(' | '.join(to_print))
    return metrics


if __name__ == '__main__':
    main()