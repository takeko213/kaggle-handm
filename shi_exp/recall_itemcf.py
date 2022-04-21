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

from utils import Logger, evaluate 



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
def recall(df_test_part, item_sim, user_item_dict, worker_id):
    """
    :param df_test_part:
    :param item_sim:
    :param user_item_dict:
    :param worker_id:
    :return:
    """
    data_list = []

    for user_id, item_id in tqdm(df_test_part[['customer_id', 'article_id']].values):
        rank = {}

        interacted_items = user_item_dict[user_id]
        # final 2 action find sim item
        interacted_items = interacted_items[::-1][:2]

        for loc, item in enumerate(interacted_items):
            # item top200
            for relate_item, wij in sorted(item_sim[item].items(), key=lambda d: d[1], reverse=True)[0:50]:
                # relate_item delete
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    # time decay
                    rank[relate_item] += wij * (0.7 ** loc)

        # get top k 
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:top_k]
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

    os.makedirs('result/itemcf_tmp', exist_ok=True)
    df_part_data.to_csv(f'result/itemcf_tmp/{worker_id}.csv', index=False)
    print(str(worker_id) + 'recall over')

def create_recall(item_sim_dict, user_items_dict, df_test, offline=True):

    df_test = df_test[['customer_id', 'article_id']].drop_duplicates()
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
        df_temp = df_test[df_test['customer_id'].isin(part_users)]
        recall(df_temp, item_sim_dict, user_items_dict, i)

    multitasking.wait_for_tasks()
    log.info('merge task')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('result/itemcf_tmp'):
        for file_name in file_list:
            df_temp = pd.read_csv(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # sort
    df_data = df_data.sort_values(['customer_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)

    # evo recall
    log.info(f'evo recall')

    total = df_test.customer_id.nunique()

    if offline:
        hitrate_5, hitrate_10, hitrate_20, hitrate_40, hitrate_50 = evaluate(
        df_data[df_data['label'].notnull()], total)

        log.debug(
            f'itemcf: hitreate5:{hitrate_5}, hitreate10:{hitrate_10}, hitreate20:{hitrate_20}, hitreate40:{hitrate_40}, hitreate50:{hitrate_50}'
        )

    # save recall result 
    df_data.to_parquet('result/recall_itemcf.parquet', index=False)

if __name__ == '__main__':
    offline = True

    INPUT_DIR = 'dataset/'
    transactions = pd.read_parquet(INPUT_DIR + 'transactions_train.parquet')

    if offline:
        transactions = transactions[(transactions.week >= transactions.week.max() - 12)  & (transactions.week < transactions.week.max())]
    else:
        transactions = transactions[transactions.week >= transactions.week.max() - 12]

    user_items_df = transactions.groupby('customer_id')['article_id'].apply(list).reset_index()
    user_items_dict = dict(zip(user_items_df['customer_id'], user_items_df['article_id']))

    with open('result/itemcf_sim.pkl', 'rb') as f:
        item_sim_dict = pickle.load(f)

    print('start itemcf recall')
    create_recall(item_sim_dict, user_items_dict, transactions, offline)
