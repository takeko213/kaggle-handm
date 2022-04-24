from unicodedata import name
import pandas as pd
import numpy as np
from utils import metrics

if __name__ == '__main__':
    test = False 
    INPUT_DIR = 'dataset/'
    if test:
        df_val = pd.read_parquet('dataset/df_val01.parquet')
    else:
        df_val = pd.read_parquet('dataset/df_val.parquet')

    df_val = pd.read_parquet('dataset/df_val.parquet')
    print(len(df_val.customer_id.unique()))  # 68984

    # df_predict = pd.read_parquet('result/recall.parquet')
    df_predict = pd.read_parquet('result/recall_itemcf.parquet')
    print(len(df_predict.customer_id.unique()))  # 505114

    df = df_predict.groupby('customer_id')['article_id'].count().reset_index()
    # print(df.head())
    print(df[df.article_id<200].head())

    df_val = df_val.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    df_val.columns = ['customer_id', 'valid_true']

    df_predict = df_predict.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    df_predict.columns = ['customer_id', 'valid_pred']

    print(df_predict.head())
    merge = df_val.merge(df_predict, how='inner')
    print(len(merge))
    metrics(merge, 200)
