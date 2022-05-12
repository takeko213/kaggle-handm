from unicodedata import name
import pandas as pd
import numpy as np
from utils import metrics, evaluate_recall

if __name__ == '__main__':
    test = False 
    INPUT_DIR = 'dataset/'
    if test:
        df_val = pd.read_parquet('dataset/df_val01.parquet')
    else:
        df_val = pd.read_parquet('dataset/df_val.parquet')

    df_val = pd.read_parquet('dataset/df_val.parquet')
    print(len(df_val.customer_id.unique()))  # 68984

    # df_predict = pd.read_parquet('result/recall_itemcf.parquet')
    # val_customers = df_val['customer_id'].unique()
    # df_predict = df_predict[df_predict.customer_id.isin(val_customers)]

    # evaluate_recall(df_predict, df_val)

    # df_predict = pd.read_parquet('result/trn_lgb_ranker_feats.parquet')
    # df = df_predict.groupby('customer_id')['article_id'].count().reset_index()
    # print(df.head())
    # print(df[df.article_id<200].head())
    
    # evaluate_recall(df_predict, df_val)
    
    df_val = df_val.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    df_val.columns = ['customer_id', 'valid_true']
    df_predict = pd.read_parquet('result/trn_lgb_ranker_feats.parquet')
    df_predict = df_predict.sort_values(['customer_id', 'pred_score'], ascending=False)
    # df_predict = df_predict.groupby('customer_id').head(12)
    df_predict = df_predict.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    df_predict.columns = ['customer_id', 'valid_pred']

    merge = df_val.merge(df_predict, how='inner')
    metrics(merge, 12)
