import argparse
from collections import defaultdict, Counter
import random

import pandas as pd
import tqdm
import numpy as np

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = Path('../input/jigsaw-unintended-bias-in-toxicity-classification')


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # df[col] = df[col].astype('category')
            pass

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def make_folds(n_folds: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    df['comment_text'] = df['comment_text'].astype(str)
    df["comment_text"] = df["comment_text"].fillna("DUMMY_VALUE")
    df = df.fillna(0)
    df['binary_target'] = (df['target'] >= 0.5).astype(float)
    df['len'] = df['comment_text'].apply(
        lambda x: len(x.split())).astype(np.int32)

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    idc2 = [
        'homosexual_gay_or_lesbian', 'jewish', 'muslim', 'black', 'white']
    # Overall
    weights = np.ones((len(df),))
    # # Subgroup
    # weights += (df[identity_columns].fillna(0).values >= 0.5).mean(axis=1) / 4
    # # Background Positive, Subgroup Negative
    # # weights += (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
    #             #  (df[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(bool).astype(np.int) / 4
    # weights += ((df['target'].values<0.5).astype(np.int)*(df[identity_columns].values>=0.5).mean(axis=1) + \
    #             (df['target'].values>=0.5).astype(np.int)*(df[identity_columns].values<0.5).mean(axis=1))/4
    # # Background Negative, Subgroup Positive
    # # weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
    # #              (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(bool).astype(np.int) / 4
    # weights += ((df['target'].values>=0.5).astype(np.int)*(df[identity_columns].values>=0.5).mean(axis=1) + \
    #             (df['target'].values<0.5).astype(np.int)*(df[identity_columns].values<0.5).mean(axis=1))/4
    weights += (df[idc2].values>=0.5).sum(axis=1).astype(np.int)
    loss_weight = 1.0 / weights.mean()
    print(weights.mean())
    df['weights'] = weights

    kf = StratifiedKFold(n_splits=n_folds, random_state=2019)
    df['fold'] = -1
    for folds, (train_index, test_index) in enumerate(kf.split(df['id'], df['binary_target'])):
        df.loc[test_index, 'fold'] = folds

    # df = reduce_mem_usage(df)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds)
    df.to_pickle(DATA_ROOT/'folds.pkl')


if __name__ == '__main__':
    main()
