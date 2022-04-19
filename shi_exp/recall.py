import numpy as np
import cudf
import os
import warnings
from collections import defaultdict
from itertools import permutations
import gc

import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')


# init log
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('multi recall merge: ')


def mms(df):
    """
    normalization
    :param df:
    :return:
    """
    user_score_max = {}
    user_score_min = {}

    # get score max min
    for user_id, g in df[['customer_id', 'sim_score']].groupby('customer_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['customer_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('customer_id')[
        'article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['customer_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('customer_id')[
        'article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['customer_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':

    recall_path = 'result'

    # recall methods
    recall_methods = ['itemcf', 'hot']
    # recall weight
    weights = {'itemcf': 1, 'hot': 0.1}
    recall_list = []
    # recall_dict = {}
    for recall_method in recall_methods:
        recall_result = pd.read_csv(
            f'{recall_path}/recall_{recall_method}.csv')
        weight = weights[recall_method]

        recall_result['sim_score'] = mms(recall_result)
        recall_result['sim_score'] = recall_result['sim_score'] * weight

        recall_list.append(recall_result)
        del recall_result 
        gc.collect()
        # recall_dict[recall_method] = recall_result

    # merge result
    recall_final = pd.concat(recall_list, sort=False)

    recall_score = recall_final[['customer_id', 'article_id', 'sim_score']].groupby(['customer_id', 'article_id'
                                                                                     ])['sim_score'].sum().reset_index()
    # drop duplicates
    recall_final = recall_final[['customer_id', 'article_id', 'label'
                                 ]].drop_duplicates(['customer_id', 'article_id'])
    # add label
    recall_final = recall_final.merge(recall_score, how='left')
    # sort with sim score
    recall_final.sort_values(
        ['customer_id', 'sim_score'], inplace=True, ascending=[True, False])
    # get recall item top 50
    recall_final = recall_final.groupby('customer_id').head(50)
    log.debug(f'recall_final.shape: {recall_final.shape}')

    # # evo recall
    # total = transactions.customer_id.nunique()
    # hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
    #     recall_final[recall_final['label'].notnull()], total)

    # log.debug(
    #     f'evo recall: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    # )

    df = recall_final['customer_id'].value_counts().reset_index()
    df.columns = ['customer_id', 'cnt']
    log.debug(f"per user recll number: {df['cnt'].mean()}")

    log.debug(
        f"label distribute: {recall_final[recall_final['label'].notnull()]['label'].value_counts()}"
    )

    recall_final.to_csv('result/recall.csv', index=False)
