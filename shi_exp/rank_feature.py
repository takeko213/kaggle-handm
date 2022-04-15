import pandas as pd
import numpy as np
import cudf
import cuml
from cuml.experimental.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os
from utils import Logger, reduce_mem

# init log
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('rank feature')


def make_article_tran_features(history):

    datediff_sale_max_min = history.groupby('article_id').agg(
        {'t_dat': ['max', 'min']}).reset_index()
    datediff_sale_max_min.columns = ['article_id', 'sale_max', 'sale_min']
    datediff_sale_max_min['datediff_sale_max_min'] = (
        datediff_sale_max_min['sale_max'] - datediff_sale_max_min['sale_min']).dt.days  # int64
    datediff_sale_max_min = datediff_sale_max_min.drop(
        columns=['sale_max', 'sale_min'])

    sale_price_total = history.groupby('article_id').agg(
        {'price': ['sum', 'max', 'min', 'mean']}).reset_index()
    sale_price_total.columns = ['article_id', 'sale_price_sum_total',
                                'sale_price_max_total', 'sale_price_min_total', 'sale_price_mean_total']

    frequency_sale_recent_total = history.groupby(
        'article_id').agg({'t_dat': ['count']}).reset_index()
    frequency_sale_recent_total.columns = [
        'article_id', 'frequency_sale_total']

    # date_condition = history.t_dat.max()-np.timedelta64(30, 'D')
    # frequency_sale_recent_1_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
    # frequency_sale_recent_1_months.columns = ['article_id','frequency_sale_recent_1_months']
    # sale_price_recent_1_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
    # sale_price_recent_1_months.columns = ['article_id', 'sale_price_sum_recent_1_months']

    # date_condition = history.t_dat.max()-np.timedelta64(60, 'D')
    # frequency_sale_recent_2_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
    # frequency_sale_recent_2_months.columns = ['article_id','frequency_sale_recent_2_months']
    # sale_price_recent_2_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
    # sale_price_recent_2_months.columns = ['article_id', 'sale_price_sum_recent_2_months']

    # date_condition = history.t_dat.max()-np.timedelta64(90, 'D')
    # frequency_sale_recent_3_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
    # frequency_sale_recent_3_months.columns = ['article_id','frequency_sale_recent_3_months']
    # sale_price_recent_3_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
    # sale_price_recent_3_months.columns = ['article_id', 'sale_price_sum_recent_3_months']

    # date_condition = history.t_dat.max()-np.timedelta64(7, 'D')
    # frequency_sale_recent_7_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
    # frequency_sale_recent_7_days.columns = ['article_id','frequency_sale_recent_7_days']
    # sale_price_recent_7_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
    # sale_price_recent_7_days.columns = ['article_id', 'sale_price_sum_recent_7_days']

    # date_condition = history.t_dat.max()-np.timedelta64(14, 'D')
    # frequency_sale_recent_14_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
    # frequency_sale_recent_14_days.columns = ['article_id','frequency_sale_recent_14_days']
    # sale_price_recent_14_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
    # sale_price_recent_14_days.columns = ['article_id', 'sale_price_sum_recent_14_days']

    # date_condition = history.t_dat.max()-np.timedelta64(21, 'D')
    # frequency_sale_recent_21_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
    # frequency_sale_recent_21_days.columns = ['article_id','frequency_sale_recent_21_days']
    # sale_price_recent_21_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
    # sale_price_recent_21_days.columns = ['article_id', 'sale_price_sum_recent_21_days']
#     print(sale_price_recent_21_days)

    dfs = [
        #            datediff_sale_max_min,
        frequency_sale_recent_total,
        #    frequency_sale_recent_1_months,
        #    frequency_sale_recent_2_months,
        #    frequency_sale_recent_3_months,
        #    frequency_sale_recent_7_days,
        #    frequency_sale_recent_14_days,
        #    frequency_sale_recent_21_days,
        sale_price_total,
        #    sale_price_recent_1_months,
        #    sale_price_recent_2_months,
        #    sale_price_recent_3_months,
        #    sale_price_recent_7_days,
        #    sale_price_recent_14_days,
        #    sale_price_recent_21_days
    ]

    result = datediff_sale_max_min
    for df in dfs:
        result = result.merge(df, on='article_id', how='left')
    result = result.fillna(0)
    normalize_columns = result.columns.tolist()
    normalize_columns.remove('article_id')

    scaler = MinMaxScaler()
    result[normalize_columns] = scaler.fit_transform(result[normalize_columns])
    return result


def make_customer_tran_features(history):

    datediff_buy_max_min = history.groupby('customer_id').agg(
        {'t_dat': ['max', 'min']}).reset_index()
    datediff_buy_max_min.columns = ['customer_id', 'buy_max', 'buy_min']
    datediff_buy_max_min['datediff_buy_max_min'] = (
        datediff_buy_max_min['buy_max'] - datediff_buy_max_min['buy_min']).dt.days  # int64
    datediff_buy_max_min = datediff_buy_max_min.drop(
        columns=['buy_max', 'buy_min'])

    buy_price_total = history.groupby('customer_id').agg(
        {'price': ['sum', 'max', 'min', 'mean']}).reset_index()
    buy_price_total.columns = ['customer_id', 'buy_price_sum_total',
                               'buy_price_max_total', 'buy_price_min_total', 'buy_price_mean_total']

    frequency_buy_recent_total = history.groupby(
        'customer_id').agg({'t_dat': ['count']}).reset_index()
    frequency_buy_recent_total.columns = ['customer_id', 'frequency_buy_total']

    # date_condition = history.t_dat.max()-np.timedelta64(30, 'D')
    # frequency_buy_recent_1_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
    # frequency_buy_recent_1_months.columns = ['customer_id','frequency_buy_recent_1_months']
    # buy_price_recent_1_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
    # buy_price_recent_1_months.columns = ['customer_id', 'buy_price_sum_recent_1_months', 'buy_price_max_recent_1_months', 'buy_price_min_recent_1_months', 'buy_price_mean_recent_1_months', 'buy_price_std_recent_1_months']

    # date_condition = history.t_dat.max()-np.timedelta64(60, 'D')
    # frequency_buy_recent_2_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
    # frequency_buy_recent_2_months.columns = ['customer_id','frequency_buy_recent_2_months']
    # buy_price_recent_2_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
    # buy_price_recent_2_months.columns = ['customer_id', 'buy_price_sum_recent_2_months', 'buy_price_max_recent_2_months', 'buy_price_min_recent_2_months', 'buy_price_mean_recent_2_months', 'buy_price_std_recent_2_months']

    # date_condition = history.t_dat.max()-np.timedelta64(90, 'D')
    # frequency_buy_recent_3_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
    # frequency_buy_recent_3_months.columns = ['customer_id','frequency_buy_recent_3_months']
    # buy_price_recent_3_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
    # buy_price_recent_3_months.columns = ['customer_id', 'buy_price_sum_recent_3_months', 'buy_price_max_recent_3_months', 'buy_price_min_recent_3_months', 'buy_price_mean_recent_3_months', 'buy_price_std_recent_3_months']

    # date_condition = history.t_dat.max()-np.timedelta64(7, 'D')
    # frequency_buy_recent_7_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
    # frequency_buy_recent_7_days.columns = ['customer_id','frequency_buy_recent_7_days']
    # buy_price_recent_7_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
    # buy_price_recent_7_days.columns = ['customer_id', 'buy_price_sum_recent_7_days', 'buy_price_max_recent_7_days', 'buy_price_min_recent_7_days', 'buy_price_mean_recent_7_days', 'buy_price_std_recent_7_days']

    # date_condition = history.t_dat.max()-np.timedelta64(14, 'D')
    # frequency_buy_recent_14_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
    # frequency_buy_recent_14_days.columns = ['customer_id','frequency_buy_recent_14_days']
    # buy_price_recent_14_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
    # buy_price_recent_14_days.columns = ['customer_id', 'buy_price_sum_recent_14_days', 'buy_price_max_recent_14_days', 'buy_price_min_recent_14_days', 'buy_price_mean_recent_14_days', 'buy_price_std_recent_14_days']

    # date_condition = history.t_dat.max()-np.timedelta64(21, 'D')
    # frequency_buy_recent_21_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
    # frequency_buy_recent_21_days.columns = ['customer_id','frequency_buy_recent_21_days']
    # buy_price_recent_21_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
    # buy_price_recent_21_days.columns = ['customer_id', 'buy_price_sum_recent_21_days', 'buy_price_max_recent_21_days', 'buy_price_min_recent_21_days', 'buy_price_mean_recent_21_days', 'buy_price_std_recent_21_days']

    dfs = [
        #          datediff_buy_max_min
        frequency_buy_recent_total,
        # frequency_buy_recent_1_months,
        # frequency_buy_recent_2_months,
        # frequency_buy_recent_3_months,
        # frequency_buy_recent_7_days,
        # frequency_buy_recent_14_days,
        # frequency_buy_recent_21_days,
        buy_price_total,
        # buy_price_recent_1_months,
        # buy_price_recent_2_months,
        # buy_price_recent_3_months,
        # buy_price_recent_7_days,
        # buy_price_recent_14_days,
        # buy_price_recent_21_days,
    ]

    result = datediff_buy_max_min
    for df in dfs:
        result = result.merge(df, on='customer_id', how='left')
    result = result.fillna(0)
#     print(result.columns)

    normalize_columns = result.columns.tolist()
    normalize_columns.remove('customer_id')
    scaler = MinMaxScaler()
    result[normalize_columns] = scaler.fit_transform(result[normalize_columns])

    return result


if __name__ == '__main__':
    INPUT_DIR = 'dataset/'

    transactions = reduce_mem(pd.read_parquet(
        INPUT_DIR + 'transactions.parquet'))
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

    # transactions['article_id'] = transactions.article_id.astype('int32')
    # transactions['customer_id'] = transactions.customer_id.astype('int32')
    # transactions.t_dat = cudf.to_datetime(transactions.t_dat)

    transactions = transactions[(transactions.t_dat >= np.datetime64(
        '2020-08-01')) & (transactions.t_dat < np.datetime64('2020-09-16'))]

    customers = pd.read_parquet(INPUT_DIR + 'customers.parquet')
    customers['age'] = customers['age'].fillna(int(customers['age'].mean()))
    customers = reduce_mem(customers).copy()
    articles = reduce_mem(pd.read_parquet(INPUT_DIR + 'articles.parquet'))
    df_recall = reduce_mem(pd.read_csv('result/recall.csv'))

    label_encode_column = [
        'FN', 'Active', 'fashion_news_frequency', 'club_member_status', 'postal_code']
    for c in label_encode_column:
        customers[c] = customers[c].astype(str)
        # le = cuml.preprocessing.LabelEncoder()
        le = LabelEncoder()
        customers[c] = le.fit_transform(customers[c].fillna(''))


    label_encode_column = ['product_type_name', 'product_group_name', 'graphical_appearance_name',
                           'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name',
                           'index_name', 'index_group_name', 'section_name', 'garment_group_name']

    for c in label_encode_column:
        articles[c] = articles[c].astype(str)
        # le = cuml.preprocessing.LabelEncoder()
        le = LabelEncoder()
        articles[c] = le.fit_transform(articles[c].fillna(''))

    label_encode_column.insert(0, 'article_id')
    articles = articles[label_encode_column]

    #  null count
    # FN 895050
    # Active 907576
    select_column = ['customer_id', 'club_member_status',
                     'fashion_news_frequency', 'age', 'postal_code']
    customers = customers[select_column]
    
    # use cudf to calculate
    transactions = cudf.DataFrame.from_pandas(transactions)

    article_tran_features = make_article_tran_features(transactions).to_pandas()
    customer_tran_features = make_customer_tran_features(transactions).to_pandas()
    df_recall = df_recall.merge(article_tran_features, how='left')
    df_recall = df_recall.merge(customer_tran_features, how='left')
    log.debug('start save rank feature')
    df_recall.to_csv('data/rank_train.csv', index=False)
    log.debug('over save rank feature')