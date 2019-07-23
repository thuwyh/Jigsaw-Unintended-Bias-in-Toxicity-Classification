import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = Path('../input/jigsaw-unintended-bias-in-toxicity-classification')


def make_folds(n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    df['comment_text'] = df['comment_text'].astype(str)
    df["comment_text"] = df["comment_text"].fillna("DUMMY_VALUE")
    df = df.fillna(0)
    identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    for c in identity_columns:
        df[c] = (df[c] >= 0.5).astype(bool)
    df['binary_target'] = (df['target'] >= 0.5).astype(bool)

    # Overall
    weights = np.ones((len(df),)) / 4
    # Subgroup
    weights += (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(
                     np.int)) > 1).astype(bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(
                     np.int)) > 1).astype(bool).astype(np.int) / 4

    print(weights.mean())

    df['weights'] = weights
    print(df["weights"])
    df['len'] = df['comment_text'].apply(lambda x: len(x.split())).astype(np.int32)

    # target stratify
    ########################################
    kf = StratifiedKFold(n_splits=n_folds, random_state=2019)
    df['fold'] = -1
    for folds, (train_index, test_index) in enumerate(kf.split(df['id'], df['binary_target'])):
        df.loc[test_index, 'fold'] = folds
    df.loc[(df['homosexual_gay_or_lesbian'] >= 0.5) | (df["muslim"] >= 0.5) | (df["black"] >= 0.5) | (
                df["white"] >= 0.5), "weights"] *= 1.5
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_pickle(str(DATA_ROOT) + '/folds_weight1.5.pkl')


if __name__ == '__main__':
    main()
