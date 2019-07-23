import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = Path('../input/jigsaw-unintended-bias-in-toxicity-classification')

def make_folds(n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    df['comment_text'] = df['comment_text'].astype(str)
    df["comment_text"] = df["comment_text"].fillna("DUMMY_VALUE")
    df = df.fillna(0)
    df['binary_target'] = (df['target'] >= 0.5).astype(float)
    df['len'] = df['comment_text'].apply(
        lambda x: len(x.split())).astype(np.int32)

    idc2 = [
        'homosexual_gay_or_lesbian', 'jewish', 'muslim', 'black', 'white']
    # Overall
    weights = np.ones((len(df),))

    weights += (df[idc2].values>=0.5).sum(axis=1).astype(np.int)
    loss_weight = 1.0 / weights.mean()
    print(weights.mean())
    df['weights'] = weights

    kf = StratifiedKFold(n_splits=n_folds, random_state=2019)
    df['fold'] = -1
    for folds, (train_index, test_index) in enumerate(kf.split(df['id'], df['binary_target'])):
        df.loc[test_index, 'fold'] = folds

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_pickle(DATA_ROOT/'folds.pkl')


if __name__ == '__main__':
    main()
