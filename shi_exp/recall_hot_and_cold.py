import os
import signal
from collections import defaultdict
import itertools
import pickle
import cudf
import numpy as np

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
log.info('hot recall ')


@multitasking.task
def recall_hot(df_test_part, hot_items, user_item_dict, worker_id):
    """
    :param df_test_part
    :param hot_items
    :param user_item_dict
    :param worker_id
    :return:
    """
    data_list = []

    print(str(worker_id) + 'start hot recall')
    for user_id, item_id in tqdm(df_test_part[['customer_id', 'article_id']].values):

        rank = {}

        interacted_items = user_item_dict[user_id]
        # get last 12 action find sim item
        interacted_items = interacted_items[::-1][:12]

        # get top 100
        for loc, relate_item in enumerate(hot_items[:100]):
            # relate_item can't be last buy
            if relate_item not in interacted_items:
                rank.setdefault(relate_item, 0)
                # time decay
                rank[relate_item] = 1/(loc+1)

        # recalculate score tak top 100
        sim_items = sorted(
            rank.items(), key=lambda d: d[1], reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_part = pd.DataFrame()
        df_part['article_id'] = item_ids
        df_part['sim_score'] = item_sim_scores
        df_part['customer_id'] = user_id
        df_part['label'] = 0
        df_part.loc[df_part['article_id'] == item_id, 'label'] = 1

        data_list.append(df_part)

    df_part_data = pd.concat(data_list, sort=False)

    os.makedirs('result/hot_tmp', exist_ok=True)
    df_part_data.to_csv(f'result/hot_tmp/{worker_id}.csv', index=False)
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
        part_users = all_users[i:i + n_len]
        df_temp = df[df['customer_id'].isin(part_users)]
        recall_hot(df_temp, hot_items_list, user_items_dict, i)

    multitasking.wait_for_tasks()
    # log.info('merge task')

    # df_data = pd.DataFrame()
    # for path, _, file_list in os.walk('result/hot_tmp'):
    #     for file_name in file_list:
    #         df_temp = pd.read_csv(os.path.join(path, file_name))
    #         df_data = df_data.append(df_temp)

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
    # df_data.to_csv('result/recall_hot.csv', index=False)


def create_reall_cold(df, hot_items_list):
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
        df_new_user['article_id'] = hot_items[:100]
        df_new_user['sim_score'] = [1/(score+1) for score in range(100)]
        df_new_user['customer_id'] = user_id[0]
        df_new_user['label'] = 0
        data_list.append(df_new_user)

    df_part_data = pd.concat(data_list, sort=False)
    os.makedirs('result/cold_tmp', exist_ok=True)
    df_part_data.to_csv(f'result/cold_tmp/{worker_id}.csv', index=False)
    print(str(worker_id) + 'cold over')


def merge(df):
    log.info('merge task')
    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('result/hot_tmp'):
        for file_name in file_list:
            df_temp = pd.read_csv(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    for path, _, file_list in os.walk('result/cold_tmp'):
        for file_name in file_list:
            df_temp = pd.read_csv(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # sort
    df_data = df_data.sort_values(['customer_id', 'sim_score'], ascending=[
                                  True, False]).reset_index(drop=True)

    # evo recall
    total = df.customer_id.nunique()

    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_data[df_data['label'].notnull()], total)

    log.debug(
        f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )
    # save recall result
    df_data.to_csv('result/recall_hot.csv', index=False)


if __name__ == '__main__':
    INPUT_DIR = 'dataset/'
    transactions = cudf.read_parquet(INPUT_DIR + 'transactions.parquet')
    transactions.t_dat = cudf.to_datetime(transactions.t_dat)
    transactions = transactions[(transactions.t_dat >= np.datetime64(
        '2020-08-01')) & (transactions.t_dat < np.datetime64('2020-09-16'))]

    # hot top 100
    hot_items_list = transactions['article_id'].value_counts(
    ).keys().to_arrow().to_pylist()[:100]

    customer_ids = cudf.read_parquet(
        INPUT_DIR + 'customers.parquet')[['customer_id']].to_pandas()

    # prepare for hot recall
    transactions = transactions.to_pandas()
    user_items_df = transactions.groupby(
        'customer_id')['article_id'].apply(list).reset_index()
    user_items_dict = dict(
        zip(user_items_df['customer_id'], user_items_df['article_id']))

    print('start hot recall')
    create_reall_hot(transactions, hot_items_list, user_items_dict)

    print('start code recall')
    create_reall_cold(customer_ids, hot_items_list)
    print('hot cold recall over')

    # start merge
    merge(transactions)
