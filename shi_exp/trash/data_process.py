"""
@Time : 2021/2/18 11:06 AM 
@Author : 猿小明
数据处理：根据用户点击日志，构建训练集和测试集
测试集：随机选出部分用户，每个用户最后一次点击
训练集：测试集用户去除最后一次点击 + 其他用户点击日志
说明：如果用户只有一条点击数据，该用户在训练集中未出现
"""


import os
import numpy as np
import pandas as pd
from utils_log import Logger
import cudf

# log
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info(f'process: data_process')


def data_split(df):
    """
    generate df_train
    generate df_test
    :param df: transactions
    :return:
    """
    
    # drop duplicate data 
    df = df.drop_duplicates(['customer_id', 'article_id', 't_dat'])
    
    # sort df by customer_id, t_dat 
    df = df.sort_values(by=['customer_id', 't_dat'])
    
    total_users = df['customer_id'].drop_duplicates()

    log.debug(f'total_users num: {total_users.shape}')

    df.to_csv('data/predict.csv', index=False)

    # create test dataset 
    test_users = transactions[transactions.t_dat > np.datetime64('2020-09-16')]['customer_id'].drop_duplicates()
    test_users = test_users.to_pandas()
    log.debug(f'test_users num: {test_users.shape}')

    # test user last purchase
    df_test = df.to_pandas().groupby('customer_id').tail(1)
    df_test = df_test[df_test['customer_id'].isin(test_users)]

    # df delete test user data
    df_test = cudf.DataFrame.from_pandas(df_test)
    df_train = df.append(df_test).drop_duplicates(keep=False)

    # save to file
    df_train.to_csv('data/my_train_set.csv', index=False)
    df_test.to_csv('data/my_test_set.csv', index=False)


if __name__ == '__main__':
    INPUT_DIR = 'dataset/'
    transactions = cudf.read_parquet(INPUT_DIR + 'transactions.parquet')
    transactions.t_dat = cudf.to_datetime(transactions.t_dat)
    transactions = transactions[transactions.t_dat > np.datetime64('2020-06-01')]
    data_split(transactions)