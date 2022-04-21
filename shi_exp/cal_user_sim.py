import pandas as pd  
import numpy as np
from tqdm import tqdm  
from collections import defaultdict  
import os, math, warnings, math, pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

warnings.filterwarnings('ignore')
save_path = 'result/'

def get_user_activate_degree_dict(df):
    df_ = df.groupby('customer_id')['article_id'].count().reset_index()
    
    mm = MinMaxScaler()
    df_['article_id'] = mm.fit_transform(df_[['article_id']])
    user_activate_degree_dict = dict(zip(df_['customer_id'], df_['article_id']))
    
    return user_activate_degree_dict

# {item1: [(user1, time1), (user2, time2)...]...}
def get_item_user_time_dict(df):
    def make_user_time_pair(df):
        return list(zip(df['customer_id'], df['week']))
    
    df = df.sort_values('week')
    item_user_time_df = df.groupby('article_id')['customer_id', 'week'].apply(lambda x: make_user_time_pair(x))\
                                                            .reset_index().rename(columns={0: 'user_time_list'})
    
    item_user_time_dict = dict(zip(item_user_time_df['article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict
    
def usercf_sim(df, user_activate_degree_dict):
    """
        calculate usercf
        :param df
        :param user_activate_degree_dict
        return usercf
    """
    item_user_time_dict = get_item_user_time_dict(df)
    
    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, _ in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, _ in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # user activate degree weight
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])   
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)
    
    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])
    
    # save result
    pickle.dump(u2u_sim_, open(save_path + 'usercf_u2u_sim.pkl', 'wb'))

    return u2u_sim_


if __name__ == '__main__':

    INPUT_DIR = 'dataset/'
    transactions = pd.read_parquet(INPUT_DIR + 'transactions_train.parquet')
    transactions = transactions[(transactions.week >= transactions.week.max() - 8)  & (transactions.week < transactions.week.max())]
    print(len(transactions[['customer_id']].drop_duplicates())) #439368

    user_activate_degree_dict = get_user_activate_degree_dict(transactions)
    df_val = pd.read_parquet('dataset/df_val.parquet')
    u2u_sim = usercf_sim(transactions, user_activate_degree_dict)