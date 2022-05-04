import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
from collections import Counter
from datetime import timedelta
from collections import defaultdict
import math
from utils import reduce_mem
import warnings

warnings.filterwarnings('ignore')

uid = 'customer_id'
iid = 'article_id'
color_col = 'colour_group_name'
index_name_col = 'index_name'
product_group_name_col = 'product_group_name'

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
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i] * item_cnt[j])

    return sim_item_corr, user_item_dict


def get_most_freq_next_item(user_group):
    next_items = {}
    for user in tqdm(user_group.keys()):
        items = user_group[user]
        for i, item in enumerate(items[:-1]):
            if item not in next_items:
                next_items[item] = []
            #             if item != items[i+1]:
            #                 next_items[item].append(items[i+1])
            next_items[item].append(items[i + 1])

    pred_next = {}
    for item in next_items:
        if len(next_items[item]) >= 5:
            most_common = Counter(next_items[item]).most_common()
            ratio = most_common[0][1] / len(next_items[item])
            if ratio >= 0.1:
                pred_next[item] = most_common[0][0]

    return pred_next


def most_freq_next_item(df, n=4):
    # recent n weeks
    user_group = df.loc[df[week_col] >= (df[week_col].max() - n)].groupby([uid])[iid].apply(list)
    pred_next = get_most_freq_next_item(user_group)
    user_group_dict = user_group.to_dict()
    return user_group_dict


def get_pop_item(df, df_items, n=8):
    train_pop = df.loc[df[week_col] >= (df[week_col].max() - n)]
    train_pop['diff'] = (train_pop[time_col].max() - train_pop[time_col])
    train_pop['pop_factor'] = 1 / (train_pop['diff'].dt.days + 1)
    popular_items_group = train_pop.groupby([iid])['pop_factor'].agg('sum').reset_index()
    popular_items_group = popular_items_group.merge(df_items, on=[iid], how='left')

    popular_items_group = popular_items_group.sort_values(
        [iid, index_name_col, product_group_name_col, color_col, 'pop_factor'], ascending=False)
    popular_items_group = popular_items_group.groupby([iid, index_name_col, color_col, product_group_name_col]).head(20)
    popular_items_group = popular_items_group[[iid, index_name_col, color_col, product_group_name_col, 'pop_factor']]
    return popular_items_group


def recent_purchaed_by_user(df):
    week = 1
    tmp = df[df.week > week].groupby([uid, iid])[time_col].agg('count').reset_index()
    tmp.columns = [uid, iid, 'cnt']
    train1 = df.merge(tmp, on=[uid, iid], how='left')
    train1 = train1.sort_values([time_col, 'cnt'], ascending=False)
    return train1


def get_color_rate_by_uid(df, df_items):
    tmp = df.merge(df_items, on=[iid], how='left')
    tmp_color = tmp.groupby([uid, color_col])[time_col].agg('count').reset_index()
    tmp_color.columns = [uid, color_col, 'color_cnt']
    tmp_total = tmp.groupby([uid])[time_col].agg('count').reset_index()
    tmp_total.columns = [uid, 'total_cnt']

    result = tmp_color.merge(tmp_total, on=uid, how='left')
    result['color_rate'] = result['color_cnt'] / result['total_cnt']
    result = result[[uid, color_col, 'color_rate']]
    return result

def get_index_name_rate_by_uid(df, df_items):
    tmp = df.merge(df_items, on=[iid], how='left')

    tmp_index = tmp.groupby([uid, index_name_col])[time_col].agg('count').reset_index()

    tmp_index.columns = [uid, index_name_col, 'index_cnt']
    tmp_total = tmp.groupby([uid])[time_col].agg('count').reset_index()
    tmp_total.columns = [uid, 'total_cnt']

    result = tmp_index.merge(tmp_total, on=uid, how='left')
    result['index_rate'] = result['index_cnt'] / result['total_cnt']
    result = result[[uid, index_name_col, 'index_rate']]
    return result

def get_prodcut_name_rate_by_uid(df, df_items):
    tmp = df.merge(df_items, on=[iid], how='left')

    tmp_product_name = tmp.groupby([uid, product_group_name_col])[time_col].agg('count').reset_index()
    tmp_product_name.columns = [uid, product_group_name_col, 'cnt']

    tmp_total = tmp.groupby([uid])[time_col].agg('count').reset_index()
    tmp_total.columns = [uid, 'total_cnt']
    result = tmp_product_name.merge(tmp_total, on=uid, how='left')
    result['prodcut_name_rate'] = result['cnt']/result['total_cnt']
    result = result[[uid, product_group_name_col, 'prodcut_name_rate']]
    return result

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

    if False:
        pop_items = get_pop_item(transactions, items)
        pop_items = reduce_mem(pop_items)
        pop_items.to_parquet('result/pop_items.parquet')
    if False:
        color_rate_by_uid = get_color_rate_by_uid(transactions, items)
        color_rate_by_uid = reduce_mem(color_rate_by_uid)
        color_rate_by_uid.to_parquet('result/color_rate_by_uid.parquet')
    if False:
        index_group_rate_by_uid = get_index_name_rate_by_uid(transactions, items)
        index_group_rate_by_uid = reduce_mem(index_group_rate_by_uid)
        index_group_rate_by_uid.to_parquet('result/index_group_rate_by_uid.parquet')
    if False:
        prodcut_name_rate_by_uid = get_prodcut_name_rate_by_uid(transactions, items)
        prodcut_name_rate_by_uid = reduce_mem(prodcut_name_rate_by_uid)
        prodcut_name_rate_by_uid.to_parquet('result/prodcut_name_rate_by_uid.parquet')

    pop_items = pd.read_parquet('result/pop_items.parquet')
    color_rate_by_uid = pd.read_parquet('result/color_rate_by_uid.parquet')
    index_group_rate_by_uid = pd.read_parquet('result/index_group_rate_by_uid.parquet')
    prodcut_name_rate_by_uid = pd.read_parquet('result/prodcut_name_rate_by_uid.parquet')
    print(color_rate_by_uid.dtypes)

    df_rate = color_rate_by_uid.merge(index_group_rate_by_uid, on=[uid], how='left')
    
    # print(items[index_name_col].unique())
    # # age bin
    # bin_list = [-1, 19, 29, 39, 49, 59, 69, 119]
    # users['age_bins'] = pd.cut(users['age'], bin_list)
    # print(users.head())

    # sim, _  = get_sim_item(transactions, uid, iid)
    # print(sim[0])

    # recent_purchaed_by_user(transactions)
