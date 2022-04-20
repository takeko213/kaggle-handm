import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

save_path = 'result/'


def get_user_item_time(click_df):
    
    click_df = click_df.sort_values('week')
    
    def make_item_time_pair(df):
        return list(zip(df['article_id'], df['week']))
    
    user_item_time_df = click_df.groupby('customer_id')['article_id', 'week'].apply(lambda x: make_item_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['customer_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict

def itemcf_sim(df):
    
    user_item_time_dict = get_user_item_time(df)
    
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # time decay 
        for loc1, (i, i_week) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_week) in enumerate(item_time_list):
                if(i == j):
                    continue 
                time_diff_weight = np.exp(0.7 ** np.abs(i_week - j_week))
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += time_diff_weight / math.log(len(item_time_list) + 1)
     
    # too hot decay 
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))
    
    return i2i_sim_


if __name__ == '__main__':

    INPUT_DIR = 'dataset/'
    transactions = pd.read_parquet(INPUT_DIR + 'transactions_train.parquet')
    transactions = transactions[(transactions.week >= transactions.week.max() - 24)  & (transactions.week < transactions.week.max())]
    print(len(transactions[['customer_id']].drop_duplicates()))

    u2u_sim = itemcf_sim(transactions)