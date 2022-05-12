import os
import signal
from collections import defaultdict
import itertools
import pickle
import cudf
import gc
import numpy as np
import sys

import multitasking
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

# log init
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('hot recall & cold recall')

# memory
top_k = 100


@multitasking.task
def recall_hot(hot_items, user_item_dict, worker_id):
    """
    :param df_test_part
    :param hot_items
    :param user_item_dict
    :param worker_id
    :return:
    """
    data_list = []

    print(str(worker_id) + 'start hot recall')
    for user_id in tqdm(user_item_dict):

        rank = {}

        interacted_items = user_item_dict[user_id]
        # get last 12 action find sim item
        interacted_items = interacted_items[::-1][:5]

        # get top k 
        for loc, relate_item in enumerate(hot_items[:top_k]):
            # relate_item can't be last buy
            if relate_item not in interacted_items:
                rank.setdefault(relate_item, 0)
                # time decay
                rank[relate_item] = 1/(loc+1)

        # recalculate score tak topk
        sim_items = sorted(
            rank.items(), key=lambda d: d[1], reverse=True)[:top_k]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_part = pd.DataFrame()
        df_part['article_id'] = item_ids
        df_part['sim_score'] = item_sim_scores
        df_part['customer_id'] = user_id

        # reduce memory
        df_part['article_id'] = df_part['article_id'].astype(np.int32)
        df_part['sim_score'] = df_part['sim_score'].astype(np.float32)
        df_part['customer_id'] = df_part['customer_id'].astype(np.int32)
        # df_part['label'] = 0
        # df_part.loc[df_part['article_id'] == item_id, 'label'] = 1

        data_list.append(df_part)

    df_part_data = pd.concat(data_list, sort=False)

    os.makedirs('result/hot_tmp', exist_ok=True)
    df_part_data.to_parquet(f'result/hot_tmp/{worker_id}.parquet', index=False)
    print(str(worker_id) + 'hot over')


def create_reall_hot(df, hot_items_list, user_items_dict):
    # recall by thread
   
    n_split = max_threads
    all_users = df['customer_id'].unique()
    total = len(all_users)
    n_len = total // n_split

    # delete file save temp task result
    for path, _, file_list in os.walk('result/hot_tmp'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_user_items_dict = dict(itertools.islice(user_items_dict.items(), i, i+n_len))
        # part_users = all_users[i:i + n_len]
        # df_temp = df[df['customer_id'].isin(part_users)]
        recall_hot(hot_items_list, part_user_items_dict, i)

    multitasking.wait_for_tasks()
    log.info('merge task')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('result/cold_tmp'):
        for file_name in file_list:
            df_temp = pd.read_parquet(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # # sort
    # df_data = df_data.sort_values(['customer_id', 'sim_score'], ascending=[
    #                               True, False]).reset_index(drop=True)

    # # evo recall
    # total = df.customer_id.nunique()

    # hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
    #     df_data[df_data['label'].notnull()], total)

    # log.debug(
    #     f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    # )
    # # save recall result
    df_data.to_parquet('result/recall_hot.pariquet', index=False)


def create_reall_cold(df, hot_items_list):
    # recall by thread
    n_split = max_threads
    all_users = df['customer_id'].unique()
    total = len(all_users)
    n_len = total // n_split

    # delete file save temp task result
    for path, _, file_list in os.walk('result/cold_tmp'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df[df['customer_id'].isin(part_users)]
        reall_cold(df_temp, hot_items_list, i)

    multitasking.wait_for_tasks()


@multitasking.task
def reall_cold(df, hot_items, worker_id):
    print(str(worker_id) + 'start cold over')
    data_list = []
    for user_id in tqdm(df[['customer_id']].values):
        df_new_user = pd.DataFrame()
        #get topk
        df_new_user['article_id'] = hot_items[:top_k]
        df_new_user['sim_score'] = [1/(score+1) for score in range(top_k)]
        df_new_user['customer_id'] = user_id[0]
        # reduce memory
        df_new_user['article_id'] = df_new_user['article_id'].astype(np.int32)
        df_new_user['sim_score'] = df_new_user['sim_score'].astype(np.float32)
        df_new_user['customer_id'] = df_new_user['customer_id'].astype(np.int32)
        # df_new_user['label'] = 0
        data_list.append(df_new_user)

    df_part_data = pd.concat(data_list, sort=False)
    os.makedirs('result/cold_tmp', exist_ok=True)
    df_part_data.to_parquet(f'result/cold_tmp/{worker_id}.parquet', index=False)
    print(str(worker_id) + 'cold over')


def merge(df):
    log.info('merge task')
    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('result/hot_tmp'):
        for file_name in file_list:
            df_temp = pd.read_parquet(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # # evo recall
    # total = df.customer_id.nunique()

    # hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
    #     df_data[df_data['label'].notnull()], total)

    # log.debug(
    #     f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    # )

    for path, _, file_list in os.walk('result/cold_tmp'):
        for file_name in file_list:
            df_temp = pd.read_parquet(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)
            del df_temp
            gc.collect()

    # sort
    df_data = df_data.sort_values(['customer_id', 'sim_score'], ascending=[
                                  True, False]).reset_index(drop=True)

    
    print(len(df_data.customer_id.unique()))
    # save recall result
    df_data.to_parquet('result/recall_hot_and_cold.parquet', index=False)


if __name__ == '__main__':
    offline = True
    test = True
    start_week =12
    INPUT_DIR = 'dataset/'
    if test:
        transactions = pd.read_parquet(INPUT_DIR + 'transactions_train_sample01.parquet')
    else:
        transactions = pd.read_parquet(INPUT_DIR + 'transactions_train.parquet')
    if offline:
        transactions = transactions[(transactions.week >= transactions.week.max(
        ) - 16) & (transactions.week < transactions.week.max())]
    else:
        transactions = transactions[transactions.week >=
                                    transactions.week.max() - start_week]
    # hot top 100
    hot_items_list = transactions['article_id'].value_counts(
    ).keys()[:100]
  
    # prepare for hot recall
    transactions = transactions[['customer_id', 'article_id']].drop_duplicates()
    user_items_df = transactions.groupby(
        'customer_id')['article_id'].apply(list).reset_index()
    user_items_dict = dict(
        zip(user_items_df['customer_id'], user_items_df['article_id']))

    # print('start cold recall')
    # create_reall_hot(transactions, hot_items_list, user_items_dict)
    # print('hots recall over')

    # cold recall
    customer_ids = pd.read_parquet(
    INPUT_DIR + 'customers.parquet')[['customer_id']]
    hot_user_ids = transactions['customer_id'].drop_duplicates()
    cold_user_ids = customer_ids[~customer_ids.customer_id.isin(hot_user_ids)]
    create_reall_cold(cold_user_ids, hot_items_list)
    print('cold recall over')

    # # start merge
    merge(transactions)