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
import cudf
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

top_k = 30


@multitasking.task
def recall(item_sim, user_item_dict, worker_id):
    """
    :param df_test_part:
    :param item_sim:
    :param user_item_dict:
    :param worker_id:
    :return:
    """
    data_list = []

    for user_id in tqdm(user_item_dict):
        rank = {}

        interacted_items = user_item_dict[user_id]
        # final 2 action find sim item
        # interacted_items = interacted_items[::-1][:10]

        for loc, item in enumerate(interacted_items):
            # at least 50
            for relate_item, wij in sorted(item_sim[item].items(), key=lambda d: d[1], reverse=True)[0:50]:
                # relate_item delete
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    # time decay
                    rank[relate_item] += wij * (0.7 ** loc)

        # get top k
        sim_items = sorted(
            rank.items(), key=lambda d: d[1], reverse=True)[:top_k]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_part = pd.DataFrame()
        df_part['article_id'] = item_ids
        df_part['sim_score'] = item_sim_scores
        df_part['customer_id'] = user_id
        # df_part['label'] = 0
        # df_part.loc[df_part['article_id'] == item_id, 'label'] = 1

        # reduce memory
        df_part['article_id'] = df_part['article_id'].astype(np.int32)
        df_part['sim_score'] = df_part['sim_score'].astype(np.float32)
        df_part['customer_id'] = df_part['customer_id'].astype(np.int32)
        # df_part['label'] = df_part['label'].astype(np.int8)

        data_list.append(df_part)

    df_part_data = pd.concat(data_list, sort=False)

    os.makedirs('result/itemcf_tmp', exist_ok=True)
    df_part_data.to_parquet(
        f'result/itemcf_tmp/{worker_id}.parquet', index=False)
    print(str(worker_id) + 'recall over')


def create_recall(item_sim_dict, user_items_dict, df_test, offline=True):

    # df_test = df_test[['customer_id', 'article_id']].drop_duplicates(keep='last')
    all_users = df_test['customer_id'].unique()
    n_split = max_threads
    total = len(all_users)
    n_len = total // n_split

    # save temp result
    for path, _, file_list in os.walk('result/itemcf_tmp'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        part_user_items_dict = dict(itertools.islice(
            user_items_dict.items(), i, i+n_len))
        # df_temp = df_test[df_test['customer_id'].isin(part_users)]

        recall(item_sim_dict, part_user_items_dict, i)

    multitasking.wait_for_tasks()
    log.info('merge task')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('result/itemcf_tmp'):
        for file_name in file_list:
            df_temp = pd.read_parquet(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # sort
    df_data = df_data.sort_values(['customer_id', 'sim_score'], ascending=[
                                  True, False]).reset_index(drop=True)

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
    df_data.to_parquet('result/recall_itemcf.parquet', index=False)


if __name__ == '__main__':
    offline = True
    start_week =12
    INPUT_DIR = 'dataset/'
    transactions = pd.read_parquet(INPUT_DIR + 'transactions_train.parquet')

    if offline:
        transactions = transactions[(transactions.week >= transactions.week.max(
        ) - start_week) & (transactions.week < transactions.week.max())]
    else:
        transactions = transactions[transactions.week >=
                                    transactions.week.max() - start_week]
    

    user_items_df = transactions.groupby(['customer_id', 'article_id'])['t_dat'].count().reset_index()
    user_items_df = user_items_df.rename(columns={'t_dat':'cnt'})
    user_items_df = user_items_df.sort_values(['customer_id', 'cnt'], ascending=False)
    user_items_df = user_items_df.groupby('customer_id').head(10)

    user_items_df = user_items_df.groupby(
        'customer_id')['article_id'].apply(list).reset_index()
    user_items_dict = dict(
        zip(user_items_df['customer_id'], user_items_df['article_id']))

    with open('result/itemcf_i2i_sim.pkl', 'rb') as f:
        item_sim_dict = pickle.load(f)

    print('start itemcf recall')
    create_recall(item_sim_dict, user_items_dict, transactions, offline)
