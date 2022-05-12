import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os, math, warnings, math, pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import sys

save_path = 'result/'


def get_user_item_time(df):
    df = df.sort_values('week')

    def make_item_time_pair(df):
        return list(zip(df['article_id'], df['week']))

    user_item_time_df = df.groupby('customer_id')['article_id', 'week'].apply(lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})

    user_item_time_dict = dict(zip(user_item_time_df['customer_id'], user_item_time_df['item_time_list']))

    #    customer_id                                     item_time_list
    # 0            0                                     [(16023, 102)]
    # 1            1                                      [(87145, 94)]
    # 2            2                                     [(78503, 103)]
    return user_item_time_dict


def itemcf_sim(df):
    print('item cf calculate start===')
    user_item_time_dict = get_user_item_time(df)

    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # time decay 
        for loc1, (i, i_week) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_week) in enumerate(item_time_list):
                if (i == j):
                    continue
                time_diff_weight = np.exp(0.9 ** np.abs(i_week - j_week))
                time_diff_weight = 1
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += time_diff_weight / math.log(len(item_time_list) + 1)

    # too hot decay 
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    for item in i2i_sim_:
        i2i_sim_[item] = sorted(i2i_sim_[item].items(), key=lambda d: d[1], reverse=True)[:50] 
        
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

    return i2i_sim_


if __name__ == '__main__':
    offline = True
    test = False 
    start_week = 52 
    INPUT_DIR = 'dataset/'
    if test:
        transactions = pd.read_parquet(INPUT_DIR + 'transactions_train_sample01.parquet')
    else:
        transactions = pd.read_parquet(INPUT_DIR + 'transactions_train.parquet')
    if offline:
        transactions = transactions[
            (transactions.week >= transactions.week.max() - start_week) & (transactions.week < transactions.week.max())]
        print(len(transactions[['customer_id']].drop_duplicates()))
    else:
        transactions = transactions[transactions.week >= transactions.week.max() - start_week]

    transactions = transactions.sort_values('week')

    print(transactions.head())

    u2u_sim = itemcf_sim(transactions)
