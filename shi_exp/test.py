import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
from collections import Counter
from datetime import timedelta
from collections import defaultdict
import math

import warnings
warnings.filterwarnings('ignore')


uid = 'customer_id'
iid = 'article_id'
time_col = 't_dat'
week_col = 'week'

def get_sim_item(df, user_col, item_col, use_iif=False):
    df = df.groupby(['customer_id']).tail(2)
    user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in tqdm(user_item_dict.items()):
        for i in items:
            item_cnt[i] += 1
            sim_item.setdefault(i, {})
            for relate_item in items:
                if i == relate_item:
                    continue
                sim_item[i].setdefault(relate_item, 0)
                if not use_iif:
                    sim_item[i][relate_item] += 1
                else:
                    sim_item[i][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij/math.sqrt(item_cnt[i]*item_cnt[j])
  
    return sim_item_corr, user_item_dict

def get_most_freq_next_item(user_group):
    next_items = {}
    for user in tqdm(user_group.keys()):
        items = user_group[user]
        for i,item in enumerate(items[:-1]):
            if item not in next_items:
                next_items[item] = []
#             if item != items[i+1]:
#                 next_items[item].append(items[i+1])
            next_items[item].append(items[i+1])
    
    pred_next = {}
    for item in next_items:
        if len(next_items[item]) >= 5:
            most_common = Counter(next_items[item]).most_common()
            ratio = most_common[0][1]/len(next_items[item])
            if ratio >= 0.1:
                pred_next[item] = most_common[0][0]
            
    return pred_next

def most_freq_next_item(df, n=4):
    # recent n weeks
    user_group = df.loc[df[week_col]>= (df[week_col].max()-n)].groupby([uid])[iid].apply(list)
    pred_next = get_most_freq_next_item(user_group)
    user_group_dict = user_group.to_dict()
    return user_group_dict

def pop_item(df, n=8):
    # recent n week transactions start
    train_pop = df.loc[df[week_col]>= (df[week_col].max()-n)]
    train_pop['diff'] = (train_pop[time_col].max() - train_pop[time_col])
    train_pop['pop_factor'] = 1 / (train_pop['diff'].dt.days + 1)
    popular_items_group = train_pop.groupby([iid])['pop_factor'].sum()
    _, popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])

    return popular_items


def recent_purchaed_by_user(df):
    week = 1
    tmp = df[df.week>week].groupby([uid,iid])[time_col].agg('count').reset_index()
    tmp.columns = [uid,iid,'cnt']
    train1 = df.merge(tmp, on = [uid,iid], how='left')
    train1 = train1.sort_values([time_col, 'cnt'],ascending=False)
    train1.index = range(len(train1))
    positive_items_per_user1 = train1.groupby([uid])[iid].apply(list)
    print(positive_items_per_user1.head())

    # df

if __name__ == '__main__':
    test = True
    if test:
        percentage = '1'
        transactions = pd.read_parquet('dataset/transactions_train_sample1.parquet')
        items = pd.read_parquet('dataset/articles_train_sample1.parquet')
        users = pd.read_parquet('dataset/customers_sample01.parquet')
    else:
        transactions = pd.read_parquet('dataset/transactions_train.parquet')
        items = pd.read_parquet('dataset/articles.parquet')
        users = pd.read_parquet('dataset/customers.parquet')

    # check data type
    # print(transactions.dtypes)

    # _pop_items = pop_item(transactions)
    # print(len(_pop_items))
    # recent_purchaed_by_user(transactions)
    # print(users.head())

    # # age bin
    # bin_list = [-1, 19, 29, 39, 49, 59, 69, 119]
    # users['age_bins'] = pd.cut(users['age'], bin_list)
    # print(users.head())


    # sim, _  = get_sim_item(transactions, uid, iid)
    # print(sim[0])

    recent_purchaed_by_user(transactions)
