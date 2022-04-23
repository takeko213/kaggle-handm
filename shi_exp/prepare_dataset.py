import pandas as pd
import cudf
import os
# helper functions
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
# from average_precision import apk
# import cuml

from utils import Logger, reduce_mem

# init log
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('preprocess dataset')

# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635
def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)

def hex_id_to_int(str):
    return int(str[-16:], 16)

if __name__ == '__main__':
    INPUT_DIR = '/home/shi/workspace/h-and-m-personalized-fashion-recommendations/'

    transactions = pd.read_csv(INPUT_DIR + 'transactions_train.csv', dtype={"article_id": "str"})
    articles = pd.read_csv(INPUT_DIR + 'articles.csv', dtype={"article_id": "str"})
    customers = pd.read_csv(INPUT_DIR + 'customers.csv')

    ALL_CUSTOMER = customers['customer_id'].unique().tolist()
    ALL_ARTICLE = articles['article_id'].unique().tolist()
    ALL_CUSTOMER.sort()
    ALL_ARTICLE.sort() 
    customer_ids = dict(list(enumerate(ALL_CUSTOMER)))
    article_ids = dict(list(enumerate(ALL_ARTICLE)))

    customer_map = {u: uidx for uidx, u in customer_ids.items()}
    article_map = {i: iidx for iidx, i in article_ids.items()}

    with open("dataset/customer_ids.pickle", 'wb') as handle:
        pickle.dump(customer_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("dataset/article_ids.pickle", 'wb') as handle:
        pickle.dump(article_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    log.info('start prepare transactions')

    transactions.t_dat = pd.to_datetime(transactions.t_dat, format='%Y-%m-%d')
    transactions['week'] = 104 - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7
    transactions.article_id = transactions['article_id'].map(article_map)
    transactions.customer_id = transactions['customer_id'].map(customer_map)
    transactions = reduce_mem(transactions)
    log.info('make sure transactions dtypes')
    log.info(transactions.dtypes)
    log.info('over prepare transactions')

    
    log.info('start prepare articles')
    articles = pd.read_csv(INPUT_DIR + 'articles.csv', dtype={"article_id": "str"})
    articles.article_id = articles['article_id'].map(article_map)
    # todo None none
    label_encode_column = ['product_type_name', 'product_group_name', 'graphical_appearance_name',
                           'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name',
                           'index_name', 'index_group_name', 'section_name', 'garment_group_name']

    for c in label_encode_column:
        articles[c] = articles[c].astype(str)
        le = LabelEncoder()
        articles[c] = le.fit_transform(articles[c].fillna(''))

    label_encode_column.insert(0, 'article_id')
    articles = articles[label_encode_column]
    articles = reduce_mem(articles)
    log.info(articles.dtypes)
    log.info('over prepare articles') 


    log.info('start prepare customers')
    customers = pd.read_csv(INPUT_DIR + 'customers.csv')
    customers.customer_id = customers['customer_id'].map(customer_map)

    # fill age with mean
    customers['age'] = customers['age'].fillna(int(customers['age'].mean()))

    # replace None to NONE
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].str.replace('None','NONE')

    label_encode_column = [
        'FN', 'Active', 'fashion_news_frequency', 'club_member_status', 'postal_code']
    for c in label_encode_column:
        customers[c] = customers[c].astype(str)
        le = LabelEncoder()
        customers[c] = le.fit_transform(customers[c].fillna(''))
        
    #  null count
    # FN 895050
    # Active 907576
    select_column = ['customer_id', 'club_member_status',
                     'fashion_news_frequency', 'age', 'postal_code']
    customers = customers[select_column]
    customers = reduce_mem(customers)
    log.info(customers.dtypes)
    log.info('over prepare customers')
    
    transactions.to_parquet('dataset/transactions_train.parquet')
    customers.to_parquet('dataset/customers.parquet')
    articles.to_parquet('dataset/articles.parquet')

    print('create val df')
    df_val = transactions[transactions.week == transactions.week.max()]
    df_val = df_val[['customer_id', 'article_id']].drop_duplicates().reset_index(drop=True)
    df_val.to_parquet('dataset/df_val.parquet')
    print('create val df over')

    for sample_repr, sample in [("01", 0.001), ("1", 0.01), ("5", 0.05)]:
        print(sample)
        customers_sample = customers.sample(int(customers.shape[0]*sample), replace=False)
        customers_sample_ids = set(customers_sample["customer_id"])
        transactions_sample = transactions[transactions["customer_id"].isin(customers_sample_ids)]
        articles_sample_ids = set(transactions_sample["article_id"])
        articles_sample = articles[articles["article_id"].isin(articles_sample_ids)]
        customers_sample.to_parquet(f"dataset/customers_sample{sample_repr}.parquet", index=False)
        transactions_sample.to_parquet(f"dataset/transactions_train_sample{sample_repr}.parquet", index=False)
        articles_sample.to_parquet(f"dataset/articles_train_sample{sample_repr}.parquet", index=False)

        df_val = transactions[transactions.week == transactions.week.max()]
        df_val = df_val[['customer_id', 'article_id']].drop_duplicates().reset_index(drop=True)
        df_val.to_parquet(f'dataset/df_val{sample_repr}.parquet')