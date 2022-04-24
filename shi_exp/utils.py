from heapq import merge
import logging
from tqdm import tqdm
import time
import pandas as pd
import numpy as np


def apk(actual, predicted, k=12):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def metrics(merged, topk=12):
    # train_unq = train_df.groupby('customer_id')[
    #     'article_id'].apply(list).reset_index()
    # train_unq.columns = ['customer_id', 'valid_pred']

    # valid_unq = val_df.groupby('customer_id')[
    #     'article_id'].apply(list).reset_index()
    # valid_unq.columns = ['customer_id', 'valid_true']

    # merged = valid_unq.merge(train_unq, how='left').fillna([''])
    # merged = merged[merged['valid_true'] != ''].reset_index(drop=True)

    score = np.mean([apk(a, p, topk) for a, p in zip(
        merged['valid_true'], merged['valid_pred'])])
    print(score)
    return score

# https://www.kaggle.com/code/baekseungyun/evaluate-how-well-you-generate-the-candidate
# https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/314458
def _evaluate_recall(actual, predict):
    act_tot = len(actual)
    pre_tot = len(predict)
    correct = actual.merge(predict, on=['customer_id', 'article_id'], how='inner').shape[0]

    print(f"[+] Recall = {correct/act_tot*100:.1f}% ({correct}/{act_tot})")
    print(f"[+] Multiple Factor = {pre_tot//correct} ({pre_tot}/{correct})")

def evaluate_recall(recall_df, df_val, topk=50):
    val_customers = df_val['customer_id'].unique()
    recall_df = recall_df[recall_df.customer_id.isin(val_customers)]
    _evaluate_recall(df_val, recall_df) 

def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
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
    end_mem = df.memory_usage().sum() / 1024 ** 2
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
