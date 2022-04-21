from heapq import merge
import logging
from tqdm import tqdm
import time
import pandas as pd
import numpy as np


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    return score / min(len(actual), k)


def metrics(train_df, val_df, topk=12):

    train_unq = train_df.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    train_unq.columns = ['customer_id', 'valid_pred']

    valid_unq = val_df.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    valid_unq.columns = ['customer_id', 'valid_true']

    merged = valid_unq.merge(train_unq, how='left').fillna([''])
    merged = merged[merged['valid_true'] != ''].reset_index(drop=True)

    score = np.mean([apk(a, p, topk) for a, p in zip(
        merged['valid_true'], merged['valid_pred'])])
    print(score)
    return score

def evaluate(df, total):
    hitrate_5 = 0
    mrr_5 = 0

    hitrate_10 = 0
    mrr_10 = 0

    hitrate_20 = 0
    mrr_20 = 0

    hitrate_40 = 0
    mrr_40 = 0

    hitrate_50 = 0
    mrr_50 = 0

    gg = df.groupby(['customer_id'])

    for _, g in tqdm(gg):
        try:
            item_id = g[g['label'] == 1]['article_id'].values[0]
        except Exception as e:
            continue

        predictions = g['article_id'].values.tolist()

        rank = 0
        while predictions[rank] != item_id:
            rank += 1

        if rank < 5:
            mrr_5 += 1.0 / (rank + 1)
            hitrate_5 += 1

        if rank < 10:
            mrr_10 += 1.0 / (rank + 1)
            hitrate_10 += 1

        if rank < 20:
            mrr_20 += 1.0 / (rank + 1)
            hitrate_20 += 1

        if rank < 40:
            mrr_40 += 1.0 / (rank + 1)
            hitrate_40 += 1

        if rank < 50:
            mrr_50 += 1.0 / (rank + 1)
            hitrate_50 += 1

    hitrate_5 /= total

    hitrate_10 /= total

    hitrate_20 /= total

    hitrate_40 /= total
    mrr_40 /= total

    hitrate_50 /= total

    return hitrate_5, hitrate_10, hitrate_20, hitrate_40, hitrate_50 


def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                # elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                #     df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    # print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
    #                                                                                                        100*(start_mem-end_mem)/start_mem,
    #                                                                                                        (time.time()-starttime)/60))
    return df


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(
        self,
        filename,
        level='debug',
        fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    ):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)

        th = logging.FileHandler(filename=filename, encoding='utf-8', mode='a')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)
