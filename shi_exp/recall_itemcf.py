from inspect import trace
import math
import os
import pathlib
import signal
from collections import defaultdict
import pickle

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import itertools

from utils import Logger, evaluate, reduce_mem


max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

# init log
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('itemcf recall ')

top_k = 500



@multitasking.task
def recall(item_sim, part_user_items_dict, hot_list, worker_id):
    """
    :param df_test_part:
    :param item_sim:
    :param user_item_dict:
    :param worker_id:
    :return:
    """
    data_list = []
    print(len(part_user_items_dict))
    for user_id in tqdm(part_user_items_dict):
    #     rank = {}

    #     interacted_items = part_user_items_dict[user_id]
    #     # final 2 action find sim item
    #     # interacted_items = interacted_items[::-1][:10]

    #     # for loc, item in enumerate(interacted_items):
    #     #     # at least 50
    #     #     for relate_item, wij in  item_sim[item][:200]:
    #     #     #for relate_item, wij in sorted(item_sim[item].items(), key=lambda d: d[1], reverse=True)[0:50]:
    #     #         # if relate_item not in interacted_items:
    #     #         # relate_item delete is not good
    #     #         if True: 
    #     #             rank.setdefault(relate_item, 0)
    #     #             # time decay
    #     #             rank[relate_item] += wij * (0.9 ** loc)

    #     # # get top k
    #     # sim_items = sorted(
    #     #     rank.items(), key=lambda d: d[1], reverse=True)[:top_k]

    #     # item_ids = [item[0] for item in sim_items]
    #     # item_sim_scores = [item[1] for item in sim_items]
        
        sim_items = part_user_items_dict[user_id]
        item_ids = sim_items
        # item_sim_scores = []
        if len(sim_items) < top_k:
            for i, item in enumerate(hot_list):
                if item in sim_items:
                    continue
                item_ids.append(item)
                # item_sim_scores.append(- i - 100)
            if len(sim_items) == top_k:
                break


        df_part = pd.DataFrame()
        df_part['article_id'] = item_ids
        # df_part['sim_score'] = item_sim_scores
        df_part['customer_id'] = user_id

        # reduce memory
        df_part['article_id'] = df_part['article_id'].astype(np.int32)
        # df_part['sim_score'] = df_part['sim_score'].astype(np.float32)
        df_part['customer_id'] = df_part['customer_id'].astype(np.int32)
        df_part = df_part.drop_duplicates() 
        data_list.append(df_part)

    df_part_data = pd.concat(data_list, sort=False)

    os.makedirs('result/itemcf_tmp', exist_ok=True)
    df_part_data.to_parquet(
        f'result/itemcf_tmp/{worker_id}.parquet', index=False)
    print(str(worker_id) + 'recall over')


def create_recall(item_sim_dict, user_items_dict, df_test, hot_list, offline=True):

    df_test = df_test[['customer_id', 'article_id']].drop_duplicates(keep='last')
    all_users = df_test['customer_id'].unique()
    n_split = max_threads
    total = len(user_items_dict)
    n_len = total // n_split

    # save temp result
    for path, _, file_list in os.walk('result/itemcf_tmp'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        part_user_items_dict = dict(itertools.islice(
            user_items_dict.items(), i, i+n_len))
        recall(item_sim_dict, part_user_items_dict,hot_list, i)

    multitasking.wait_for_tasks()
    log.info('merge task')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('result/itemcf_tmp'):
        for file_name in file_list:
            df_temp = pd.read_parquet(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # sort
    # df_data = df_data.sort_values(['customer_id', 'sim_score'], ascending=[
    #                               True, False]).reset_index(drop=True)

    # evo recall
    log.info(f'evo recall')

    total = df_test.customer_id.nunique()

    # if offline:
    #     hitrate_5, hitrate_10, hitrate_20, hitrate_40, hitrate_50 = evaluate(
    #         df_data[df_data['label'].notnull()], total)

    #     log.debug(
    #         f'itemcf: hitreate5:{hitrate_5}, hitreate10:{hitrate_10}, hitreate20:{hitrate_20}, hitreate40:{hitrate_40}, hitreate50:{hitrate_50}'
    #     )

    # save recall result
    df_data = df_data.drop_duplicates()
    df_data.to_parquet('result/recall_itemcf.parquet', index=False)

#copy from t88
def get_customer_frequent(history, n=12):
    """顧客ごと商品の購入数をカウントし上位の商品を抽出

    Args:
        history (dataframe): 集計対象の実績データ
        n (int): レコメンド対象とする数
        timedelta (dateutil.relativedelta): 指定された場合、実績データの終端からtimedelta分のデータを取得する

    Returns:
        dataframe: 抽出結果
    """
        
    results = pd.DataFrame()

    # '全期間', '直近1week', '直近1month', '直近1year
    for sw in [104 , 1, 4, 52]:
        # sw start week
        customer_agg = history[history.week >=history.week.max() - sw].groupby(['customer_id', 'article_id'])['t_dat'].count().reset_index()
        customer_agg = customer_agg.rename(columns={'t_dat':'cnt'})
        customer_agg = customer_agg.sort_values(['customer_id', 'cnt'], ascending=False)
        result = customer_agg.groupby('customer_id').head(n)
        result = result[['customer_id', 'article_id']]
        results = results.append(result) 

    results = results.drop_duplicates(keep='first')
    return results


#copy from t88
def get_customer_type_frequent(history, n=12, timedelta=None):
    if timedelta is not None:
        st_date = history['t_dat'].max() - timedelta
        history = history[history['t_dat']>=st_date].copy()

    result = history[['customer_id', 'customer_type']].drop_duplicates().copy()
    agg = history.groupby(['customer_type', 'article_id'])['t_dat'].count().reset_index()
    agg = agg.rename(columns={'t_dat':'cnt'})
    agg = agg.sort_values(['customer_type', 'cnt'], ascending=False)
    agg = agg.groupby('customer_type').head(n)
    result = result.merge(agg[['customer_type', 'article_id']], on='customer_type', how='left')
    return result[['customer_id', 'article_id']]

#copy from t88
def get_article_type_frequent(history, col, n=12, timedelta=None):
    if timedelta is not None:
        st_date = history['t_dat'].max() - timedelta
        history = history[history['t_dat']>=st_date].copy()

    result = history.groupby(['customer_id', col])['t_dat'].count().reset_index()
    result = result.rename(columns={'t_dat':'cnt'})
    result = result.sort_values(['customer_id', 'cnt'], ascending=False)
    result = result.groupby(['customer_id']).head(1)[['customer_id', col]]

    agg = history.groupby([col, 'article_id'])['t_dat'].count().reset_index()
    agg = agg.rename(columns={'t_dat':'cnt'})
    agg = agg.sort_values([col, 'cnt'], ascending=False)
    agg = agg.groupby(col).head(n)
    result = result.merge(agg[[col, 'article_id']], on=col, how='left')
    return result[['customer_id', 'article_id']]

#copy from t88
def get_popular_article(history, n=12, sw=None):
    """全体の購入数をカウントし上位の商品を抽出

    Args:
        history (dataframe): 集計対象の実績データ
        n (int): レコメンド対象とする数
        timedelta (dateutil.relativedelta): 指定された場合、実績データの終端からtimedelta分のデータを取得する

    Returns:
        list: 抽出結果
    """
    result = pd.DataFrame()
    hot_items_lists = [] 
    # '全期間', '直近1week', '直近1month', '直近1year
    for sw in [104, 1, 4, 52]:
        hot_items_list = history[history.week >=history.week.max() - sw]['article_id'].value_counts().keys()[:n].to_list()
        hot_items_lists = hot_items_lists + hot_items_list
    result['article_id'] = hot_items_lists 
    result = result.drop_duplicates(keep='first')
    return hot_items_lists


if __name__ == '__main__':
    offline = True
    test = False
    start_week =104

    INPUT_DIR = 'dataset/'
    if test:
        transactions = pd.read_parquet(INPUT_DIR + 'transactions_train_sample01.parquet')
    else:
        transactions = pd.read_parquet(INPUT_DIR + 'transactions_train.parquet')

    if offline:
        transactions = transactions[(transactions.week >= transactions.week.max(
        ) - start_week) & (transactions.week < transactions.week.max())]
    else:
        transactions = transactions[transactions.week >=
                                    transactions.week.max() - start_week]
    
    hot_list = get_popular_article(transactions) 
    # print(hot_list)


    user_items_df = get_customer_frequent(transactions)
    # print(user_feq)

    user_items_df = user_items_df.groupby(
        'customer_id')['article_id'].apply(list).reset_index()
    user_items_dict = dict(
        zip(user_items_df['customer_id'], user_items_df['article_id']))

    # print(user_items_dict)
    # with open('result/itemcf_i2i_sim.pkl', 'rb') as f:
    #     item_sim_dict = pickle.load(f)
    item_sim_dict = []
    print('start itemcf recall')
    create_recall(item_sim_dict, user_items_dict, transactions, hot_list, offline)
