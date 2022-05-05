import re
import types
import builtins
import pandas as pd
import numpy as np
import cudf
import cuml
from cuml.experimental.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os
from utils import Logger, reduce_mem
import sys
from dateutil.relativedelta import relativedelta
import gc
import pickle
import itertools
from tqdm import tqdm

# init log
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('rank feature')

# def make_article_tran_features(history):
#     datediff_sale_max_min = history.groupby('article_id').agg(
#         {'t_dat': ['max', 'min']}).reset_index()
#     datediff_sale_max_min.columns = ['article_id', 'sale_max', 'sale_min']
#     datediff_sale_max_min['datediff_sale_max_min'] = (
#             datediff_sale_max_min['sale_max'] - datediff_sale_max_min['sale_min']).dt.days  # int64
#     datediff_sale_max_min = datediff_sale_max_min.drop(
#         columns=['sale_max', 'sale_min'])

#     sale_price_total = history.groupby('article_id').agg(
#         {'price': ['sum', 'max', 'min', 'mean']}).reset_index()
#     sale_price_total.columns = ['article_id', 'sale_price_sum_total',
#                                 'sale_price_max_total', 'sale_price_min_total', 'sale_price_mean_total']

#     frequency_sale_recent_total = history.groupby(
#         'article_id').agg({'t_dat': ['count']}).reset_index()
#     frequency_sale_recent_total.columns = [
#         'article_id', 'frequency_sale_total']

#     date_condition = history.t_dat.max()-np.timedelta64(30, 'D')
#     frequency_sale_recent_1_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
#     frequency_sale_recent_1_months.columns = ['article_id','frequency_sale_recent_1_months']
#     sale_price_recent_1_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
#     sale_price_recent_1_months.columns = ['article_id', 'sale_price_sum_recent_1_months']

#     date_condition = history.t_dat.max()-np.timedelta64(60, 'D')
#     frequency_sale_recent_2_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
#     frequency_sale_recent_2_months.columns = ['article_id','frequency_sale_recent_2_months']
#     sale_price_recent_2_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
#     sale_price_recent_2_months.columns = ['article_id', 'sale_price_sum_recent_2_months']

#     date_condition = history.t_dat.max()-np.timedelta64(90, 'D')
#     frequency_sale_recent_3_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
#     frequency_sale_recent_3_months.columns = ['article_id','frequency_sale_recent_3_months']
#     sale_price_recent_3_months = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
#     sale_price_recent_3_months.columns = ['article_id', 'sale_price_sum_recent_3_months']

#     date_condition = history.t_dat.max()-np.timedelta64(7, 'D')
#     frequency_sale_recent_7_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
#     frequency_sale_recent_7_days.columns = ['article_id','frequency_sale_recent_7_days']
#     sale_price_recent_7_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
#     sale_price_recent_7_days.columns = ['article_id', 'sale_price_sum_recent_7_days']

#     date_condition = history.t_dat.max()-np.timedelta64(14, 'D')
#     frequency_sale_recent_14_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
#     frequency_sale_recent_14_days.columns = ['article_id','frequency_sale_recent_14_days']
#     sale_price_recent_14_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
#     sale_price_recent_14_days.columns = ['article_id', 'sale_price_sum_recent_14_days']

#     date_condition = history.t_dat.max()-np.timedelta64(21, 'D')
#     frequency_sale_recent_21_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'t_dat':['count']}).reset_index()
#     frequency_sale_recent_21_days.columns = ['article_id','frequency_sale_recent_21_days']
#     sale_price_recent_21_days = history[history.t_dat >= date_condition].groupby('article_id').agg({'price':['sum']}).reset_index()
#     sale_price_recent_21_days.columns = ['article_id', 'sale_price_sum_recent_21_days']
#         # print(sale_price_recent_21_days)

#     dfs = [
#         #            datediff_sale_max_min,
#         frequency_sale_recent_total,
#            frequency_sale_recent_1_months,
#            frequency_sale_recent_2_months,
#            # frequency_sale_recent_3_months,
#            # frequency_sale_recent_7_days,
#            # frequency_sale_recent_14_days,
#            # frequency_sale_recent_21_days,
#         sale_price_total,
#            sale_price_recent_1_months,
#            sale_price_recent_2_months,
#            # sale_price_recent_3_months,
#            # sale_price_recent_7_days,
#            # sale_price_recent_14_days,
#            # sale_price_recent_21_days
#     ]

#     result = datediff_sale_max_min
#     for df in dfs:
#         result = result.merge(df, on='article_id', how='left')
#     result = result.fillna(0)
#     normalize_columns = result.columns.tolist()
#     normalize_columns.remove('article_id')

#     scaler = MinMaxScaler()
#     result[normalize_columns] = scaler.fit_transform(result[normalize_columns])
#     return result


# def make_customer_tran_features(history):
#     datediff_buy_max_min = history.groupby('customer_id').agg(
#         {'t_dat': ['max', 'min']}).reset_index()
#     datediff_buy_max_min.columns = ['customer_id', 'buy_max', 'buy_min']
#     datediff_buy_max_min['datediff_buy_max_min'] = (
#             datediff_buy_max_min['buy_max'] - datediff_buy_max_min['buy_min']).dt.days  # int64
#     datediff_buy_max_min = datediff_buy_max_min.drop(
#         columns=['buy_max', 'buy_min'])

#     buy_price_total = history.groupby('customer_id').agg(
#         {'price': ['sum', 'max', 'min', 'mean']}).reset_index()
#     buy_price_total.columns = ['customer_id', 'buy_price_sum_total',
#                                'buy_price_max_total', 'buy_price_min_total', 'buy_price_mean_total']

#     frequency_buy_recent_total = history.groupby(
#         'customer_id').agg({'t_dat': ['count']}).reset_index()
#     frequency_buy_recent_total.columns = ['customer_id', 'frequency_buy_total']

#     date_condition = history.t_dat.max()-np.timedelta64(30, 'D')
#     frequency_buy_recent_1_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
#     frequency_buy_recent_1_months.columns = ['customer_id','frequency_buy_recent_1_months']
#     buy_price_recent_1_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
#     buy_price_recent_1_months.columns = ['customer_id', 'buy_price_sum_recent_1_months', 'buy_price_max_recent_1_months', 'buy_price_min_recent_1_months', 'buy_price_mean_recent_1_months', 'buy_price_std_recent_1_months']

#     date_condition = history.t_dat.max()-np.timedelta64(60, 'D')
#     frequency_buy_recent_2_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
#     frequency_buy_recent_2_months.columns = ['customer_id','frequency_buy_recent_2_months']
#     buy_price_recent_2_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
#     buy_price_recent_2_months.columns = ['customer_id', 'buy_price_sum_recent_2_months', 'buy_price_max_recent_2_months', 'buy_price_min_recent_2_months', 'buy_price_mean_recent_2_months', 'buy_price_std_recent_2_months']

#     date_condition = history.t_dat.max()-np.timedelta64(90, 'D')
#     frequency_buy_recent_3_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
#     frequency_buy_recent_3_months.columns = ['customer_id','frequency_buy_recent_3_months']
#     buy_price_recent_3_months = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
#     buy_price_recent_3_months.columns = ['customer_id', 'buy_price_sum_recent_3_months', 'buy_price_max_recent_3_months', 'buy_price_min_recent_3_months', 'buy_price_mean_recent_3_months', 'buy_price_std_recent_3_months']

#     date_condition = history.t_dat.max()-np.timedelta64(7, 'D')
#     frequency_buy_recent_7_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
#     frequency_buy_recent_7_days.columns = ['customer_id','frequency_buy_recent_7_days']
#     buy_price_recent_7_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
#     buy_price_recent_7_days.columns = ['customer_id', 'buy_price_sum_recent_7_days', 'buy_price_max_recent_7_days', 'buy_price_min_recent_7_days', 'buy_price_mean_recent_7_days', 'buy_price_std_recent_7_days']

#     date_condition = history.t_dat.max()-np.timedelta64(14, 'D')
#     frequency_buy_recent_14_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
#     frequency_buy_recent_14_days.columns = ['customer_id','frequency_buy_recent_14_days']
#     buy_price_recent_14_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
#     buy_price_recent_14_days.columns = ['customer_id', 'buy_price_sum_recent_14_days', 'buy_price_max_recent_14_days', 'buy_price_min_recent_14_days', 'buy_price_mean_recent_14_days', 'buy_price_std_recent_14_days']

#     date_condition = history.t_dat.max()-np.timedelta64(21, 'D')
#     frequency_buy_recent_21_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'t_dat':['count']}).reset_index()
#     frequency_buy_recent_21_days.columns = ['customer_id','frequency_buy_recent_21_days']
#     buy_price_recent_21_days = history[history.t_dat >= date_condition].groupby('customer_id').agg({'price':['sum','max', 'min', 'mean', 'std']}).reset_index()
#     buy_price_recent_21_days.columns = ['customer_id', 'buy_price_sum_recent_21_days', 'buy_price_max_recent_21_days', 'buy_price_min_recent_21_days', 'buy_price_mean_recent_21_days', 'buy_price_std_recent_21_days']

#     dfs = [
#         #          datediff_buy_max_min
#         frequency_buy_recent_total,
#         frequency_buy_recent_1_months,
#         frequency_buy_recent_2_months,
#         # frequency_buy_recent_3_months,
#         # frequency_buy_recent_7_days,
#         # frequency_buy_recent_14_days,
#         # frequency_buy_recent_21_days,
#         buy_price_total,
#         buy_price_recent_1_months,
#         buy_price_recent_2_months,
#         # buy_price_recent_3_months,
#         # buy_price_recent_7_days,
#         # buy_price_recent_14_days,
#         # buy_price_recent_21_days,
#     ]

#     result = datediff_buy_max_min
#     for df in dfs:
#         result = result.merge(df, on='customer_id', how='left')
#     result = result.fillna(0)
#     #     print(result.columns)

#     normalize_columns = result.columns.tolist()
#     normalize_columns.remove('customer_id')
#     scaler = MinMaxScaler()
#     result[normalize_columns] = scaler.fit_transform(result[normalize_columns])

#     return result


def imports():
    for name, val in globals().items():
        # module imports
        if isinstance(val, types.ModuleType):
            yield name, val

            # functions / callables
        if hasattr(val, '__call__'):
            yield name, val


def noglobal(f):
    '''
    ref: https://gist.github.com/raven38/4e4c3c7a179283c441f575d6e375510c
    '''
    return types.FunctionType(f.__code__,
                              dict(imports()),
                              f.__name__,
                              f.__defaults__,
                              f.__closure__
                              )


@noglobal
def get_customer_frequent(history, n=12, timedelta=None):
    """顧客ごと商品の購入数をカウントし上位の商品を抽出

    Args:
        history (dataframe): 集計対象の実績データ
        n (int): レコメンド対象とする数
        timedelta (dateutil.relativedelta): 指定された場合、実績データの終端からtimedelta分のデータを取得する

    Returns:
        dataframe: 抽出結果
    """
    if timedelta is not None:
        st_date = history['t_dat'].max() - timedelta
        history = history[history['t_dat']>=st_date].copy()
        
    customer_agg = history.groupby(['customer_id', 'article_id'])['t_dat'].count().reset_index()
    customer_agg = customer_agg.rename(columns={'t_dat':'cnt'})
    customer_agg = customer_agg.sort_values(['customer_id', 'cnt'], ascending=False)
    result = customer_agg.groupby('customer_id').head(n)
    return result[['customer_id', 'article_id']]

@noglobal
def get_popular_article(history, n=12, timedelta=None):
    """全体の購入数をカウントし上位の商品を抽出

    Args:
        history (dataframe): 集計対象の実績データ
        n (int): レコメンド対象とする数
        timedelta (dateutil.relativedelta): 指定された場合、実績データの終端からtimedelta分のデータを取得する

    Returns:
        list: 抽出結果
    """
    # 全体の購入数量
    if timedelta is not None:
        st_date = history['t_dat'].max() - timedelta
        history = history[history['t_dat']>=st_date].copy()

    total_agg = history.groupby('article_id')['t_dat'].count().reset_index()
    total_agg = total_agg.rename(columns={'t_dat':'cnt'})
    total_agg = total_agg.sort_values(['cnt'], ascending=False)
    total_agg = total_agg.head(n)
    result = list(total_agg['article_id'].values)
    return result

@noglobal
def get_customer_type_frequent(history, n=12, timedelta=None):
    if timedelta is not None:
        st_date = history['t_dat'].max() - timedelta
        history = history[history['t_dat']>=st_date].copy()

    result = history[['customer_id', 'customer_type']].drop_duplicates().copy()
    agg = history.groupby(['customer_type', 'article_id'])['t_dat'].count().reset_index()
    agg = agg.rename(columns={'t_dat':'cnt'})
    agg = agg.sort_values(['customer_type', 'cnt'], ascending=False)
    agg = agg.groupby('customer_type').head(n)
    result = result.merge(agg[['customer_type', 'article_id']], on='customer_type', how='left')
    return result[['customer_id', 'article_id']]

@noglobal
def get_article_type_frequent(history, col, n=12, timedelta=None):
    if timedelta is not None:
        st_date = history['t_dat'].max() - timedelta
        history = history[history['t_dat']>=st_date].copy()

    result = history.groupby(['customer_id', col])['t_dat'].count().reset_index()
    result = result.rename(columns={'t_dat':'cnt'})
    result = result.sort_values(['customer_id', 'cnt'], ascending=False)
    result = result.groupby(['customer_id']).head(1)[['customer_id', col]]

    agg = history.groupby([col, 'article_id'])['t_dat'].count().reset_index()
    agg = agg.rename(columns={'t_dat':'cnt'})
    agg = agg.sort_values([col, 'cnt'], ascending=False)
    agg = agg.groupby(col).head(n)
    result = result.merge(agg[[col, 'article_id']], on=col, how='left')
    return result[['customer_id', 'article_id']]

@noglobal
def get_popular_new_article(first_week_sales_pred, n=12):
    """新商品の初週売り上げ予測が高い商品を抽出
    """
    first_week_sales_pred = first_week_sales_pred.sort_values(['1st_week_sales_pred'], ascending=False)
    first_week_sales_pred = first_week_sales_pred.head(n)
    result = list(first_week_sales_pred['article_id'].values)
    return result

@noglobal
def calc_pair(history):
    df = history[['article_id', 't_dat', 'customer_id']].copy()
    df = cudf.from_pandas(df)
    df['t_dat'] = df['t_dat'].factorize()[0].astype('int16')
    dt = df.groupby(['customer_id','t_dat'])['article_id'].agg(list).rename('pair').reset_index()
    df = df[['customer_id', 't_dat', 'article_id']].merge(dt, on=['customer_id', 't_dat'], how='left')
    del dt
    gc.collect()

    # Explode the rows vs list of articles
    df = df[['article_id', 'pair']].explode(column='pair')
    gc.collect()
        
    # Discard duplicates
    df = df.loc[df['article_id']!=df['pair']].reset_index(drop=True)
    gc.collect()

    # Count how many times each pair combination happens
    df = df.groupby(['article_id', 'pair']).size().rename('count').reset_index()
    gc.collect()
        
    # Sort by frequency
    df = df.sort_values(['article_id' ,'count'], ascending=False).reset_index(drop=True)
    gc.collect()

    # pick only top1 most frequent pair
    df = df.groupby('article_id').nth(0).reset_index()
    pair = dict(zip(df['article_id'].to_arrow().to_pylist(), df['pair'].to_arrow().to_pylist()))

    return pair

@noglobal
def get_reccomend(target_customer_id, history, Ns, first_week_sales_pred):
    n = 12
    result = pd.DataFrame()
    

    td = None
    result = result.append(get_customer_frequent(history, Ns['cf_a'], td))
    result = result.append(get_customer_type_frequent(history, Ns['ctf_a'], td))
    result = result.append(get_article_type_frequent(history, 'department_name', Ns['atfd_a'], td))
    result = result.append(get_article_type_frequent(history, 'perceived_colour_master_name', Ns['atfp_a'], td))
    popular_article = get_popular_article(history, Ns['pa_a'], td)
    # customerとpopular articleの全組み合わせでdataframe作成
    popular_article = pd.DataFrame(itertools.product(target_customer_id, popular_article), columns=['customer_id', 'article_id'])
    result = result.append(popular_article)

    # popular_new_article = get_popular_new_article(first_week_sales_pred, n=48)
    # popular_new_article = pd.DataFrame(itertools.product(target_customer_id, popular_new_article), columns=['customer_id', 'article_id'])
    # result = result.append(popular_new_article)

    result = result.drop_duplicates()

    td = relativedelta(weeks=1)
    result = result.append(get_customer_frequent(history, Ns['cf_w'], td))
    result = result.append(get_customer_type_frequent(history, Ns['ctf_w'], td))
    result = result.append(get_article_type_frequent(history, 'department_name', Ns['atfd_w'], td))
    result = result.append(get_article_type_frequent(history, 'perceived_colour_master_name', Ns['atfp_w'], td))
    popular_article = get_popular_article(history, Ns['pa_w'], td)
    # customerとpopular articleの全組み合わせでdataframe作成
    popular_article = pd.DataFrame(itertools.product(target_customer_id, popular_article), columns=['customer_id', 'article_id'])
    result = result.append(popular_article)
    result = result.drop_duplicates()

    td = relativedelta(months=1)
    result = result.append(get_customer_frequent(history, Ns['cf_m'], td))
    result = result.append(get_customer_type_frequent(history, Ns['ctf_m'], td))
    result = result.append(get_article_type_frequent(history, 'department_name', Ns['atfd_m'], td))
    result = result.append(get_article_type_frequent(history, 'perceived_colour_master_name', Ns['atfp_m'], td))
    popular_article = get_popular_article(history, Ns['pa_m'], td)
    # customerとpopular articleの全組み合わせでdataframe作成
    popular_article = pd.DataFrame(itertools.product(target_customer_id, popular_article), columns=['customer_id', 'article_id'])
    result = result.append(popular_article)
    result = result.drop_duplicates()

    td = relativedelta(years=1)
    result = result.append(get_customer_frequent(history, Ns['cf_y'], td))
    result = result.append(get_customer_type_frequent(history, Ns['ctf_y'], td))
    result = result.append(get_article_type_frequent(history, 'department_name', Ns['atfd_y'], td))
    result = result.append(get_article_type_frequent(history, 'perceived_colour_master_name', Ns['atfp_y'], td))
    popular_article = get_popular_article(history, Ns['pa_y'], td)
    # customerとpopular articleの全組み合わせでdataframe作成
    popular_article = pd.DataFrame(itertools.product(target_customer_id, popular_article), columns=['customer_id', 'article_id'])
    result = result.append(popular_article)
    result = result.drop_duplicates()

    result = result[result['customer_id'].isin(target_customer_id)].copy()

    purchased_together_pair = calc_pair(history)
    add_result = result.copy()
    add_result['article_id'] = add_result['article_id'].map(purchased_together_pair)
    result = result.append(add_result.dropna().drop_duplicates())
    result = result.drop_duplicates()

    return result

@noglobal
def make_article_features(articles):
    cols = ['product_type_name', 'product_group_name', 'graphical_appearance_name',
            'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name',
            'index_name', 'index_group_name', 'section_name', 'garment_group_name']
    return articles[['article_id']+cols]

@noglobal
def make_article_tran_features(history):
    df = history.groupby('article_id').agg({'t_dat':['count', 'max', 'min'],
                                            'price':['max', 'min', 'mean'], 
                                            'age':['max', 'min', 'mean', 'std']}).reset_index()
    df.columns = ['article_id','article_total_cnt', 'article_total_latest_buy', 'article_total_1st_buy', 'article_price_max', 'article_price_min', 'article_price_mean', 'article_age_max', 'article_age_min', 'article_age_mean', 'article_age_std']
    df['article_total_1st_buy'] = (history['t_dat'].max() - df['article_total_1st_buy']).dt.days
    df['article_total_latest_buy'] = (history['t_dat'].max() - df['article_total_latest_buy']).dt.days

    history_1weekago = history.loc[(history['t_dat'] > history['t_dat'].max() - relativedelta(days=7)) & 
                                   (history['t_dat'] <= history['t_dat'].max())]

    history_1weekago_df = history_1weekago.groupby('article_id').agg({'t_dat':['count', 'max', 'min'],
                                            'price':['max', 'min', 'mean'], 
                                            'age':['max', 'min', 'mean', 'std', 'median']}).reset_index()
    history_1weekago_df.columns = ['article_id','article_total_cnt_1weekago', 'article_total_latest_buy_1weekago', 'article_total_1st_buy_1weekago', 'article_price_max_1weekago', 'article_price_min_1weekago', 'article_price_mean_1weekago', 'article_age_max_1weekago', 'article_age_min_1weekago', 'article_age_mean_1weekago', 'article_age_std_1weekago', 'article_age_median_1weekago']
    history_1weekago_df['article_total_1st_buy_1weekago'] = (history_1weekago['t_dat'].max() - history_1weekago_df['article_total_1st_buy_1weekago']).dt.days
    history_1weekago_df['article_total_latest_buy_1weekago'] = (history_1weekago['t_dat'].max() - history_1weekago_df['article_total_latest_buy_1weekago']).dt.days

    df = pd.merge(df,history_1weekago_df,how='left',on='article_id')

    del history_1weekago, history_1weekago_df
    gc.collect()

    history_2weekago = history.loc[(history['t_dat'] > history['t_dat'].max() - relativedelta(days=14)) & 
                                   (history['t_dat'] <= history['t_dat'].max() - relativedelta(days=7))]

    history_2weekago_df = history_2weekago.groupby('article_id').agg({'t_dat':['count', 'max', 'min'],
                                            'price':['max', 'min', 'mean'], 
                                            'age':['max', 'min', 'mean', 'std', 'median']}).reset_index()
    history_2weekago_df.columns = ['article_id','article_total_cnt_2weekago', 'article_total_latest_buy_2weekago', 'article_total_1st_buy_2weekago', 'article_price_max_2weekago', 'article_price_min_2weekago', 'article_price_mean_2weekago', 'article_age_max_2weekago', 'article_age_min_2weekago', 'article_age_mean_2weekago', 'article_age_std_2weekago', 'article_age_median_2weekago']
    history_2weekago_df['article_total_1st_buy_2weekago'] = (history_2weekago['t_dat'].max() - history_2weekago_df['article_total_1st_buy_2weekago']).dt.days
    history_2weekago_df['article_total_latest_buy_2weekago'] = (history_2weekago['t_dat'].max() - history_2weekago_df['article_total_latest_buy_2weekago']).dt.days

    df = pd.merge(df,history_2weekago_df,how='left',on='article_id')

    del history_2weekago, history_2weekago_df
    gc.collect()

    return df


@noglobal
def make_customer_features(customers):
    return customers

@noglobal
def make_customer_tran_features(history):
    group = ['Ladieswear', 'Divided', 'Menswear', 'Sport', 'Baby/Children']
    for g in group:
        history[g] = 0
        history.loc[history['index_group_name']==g, g] = 1


    df = history.groupby('customer_id').agg({'t_dat':['count', 'max', 'min'],
                                            'price':['max', 'min', 'mean'],
                                            'Ladieswear':'sum',
                                            'Divided':'sum',
                                            'Menswear':'sum',
                                            'Sport':'sum',
                                            'Baby/Children':'sum'}).reset_index()
    df.columns = ['customer_id','customer_total_cnt', 'customer_total_latest_buy', 'customer_total_1st_buy', 
                  'customer_price_max', 'customer_price_min', 'customer_price_mean',
                  'Ladieswear', 'Divided', 'Menswear', 'Sport', 'Baby/Children']
    df['customer_total_1st_buy'] = (history['t_dat'].max() - df['customer_total_1st_buy']).dt.days
    df['customer_total_latest_buy'] = (history['t_dat'].max() - df['customer_total_latest_buy']).dt.days

    for g in group:
        df[g] = df[g] / df['customer_total_cnt']

#####################iida_exp24#########################
    history_1weekago = history.loc[(history['t_dat'] > history['t_dat'].max() - relativedelta(days=7)) & 
                                   (history['t_dat'] <= history['t_dat'].max())]
    history_1weekago_df =  history_1weekago.groupby('customer_id').agg({'t_dat':['count', 'max', 'min'],
                                            'price':['max', 'min', 'mean'],
                                            'Ladieswear':'sum',
                                            'Divided':'sum',
                                            'Menswear':'sum',
                                            'Sport':'sum',
                                            'Baby/Children':'sum'}).reset_index() 
    history_1weekago_df.columns = ['customer_id','customer_total_cnt_1weekago', 'customer_total_latest_buy_1weekago', 'customer_total_1st_buy_1weekago', 
                                'customer_price_max_1weekago', 'customer_price_min_1weekago', 'customer_price_mean_1weekago',
                                'Ladieswear_1weekago', 'Divided_1weekago', 'Menswear_1weekago', 'Sport_1weekago', 'Baby/Children_1weekago']

    history_1weekago_df['customer_total_1st_buy_1weekago'] = (history_1weekago['t_dat'].max() - history_1weekago_df['customer_total_1st_buy_1weekago']).dt.days
    history_1weekago_df['customer_total_latest_buy_1weekago'] = (history_1weekago['t_dat'].max() - history_1weekago_df['customer_total_latest_buy_1weekago']).dt.days

    for g in group:
        history_1weekago_df[g+'_1weekago'] = history_1weekago_df[g+'_1weekago'] / history_1weekago_df['customer_total_cnt_1weekago']

    df = pd.merge(df,history_1weekago_df,how='left',on='customer_id')

    history_2weekago = history.loc[(history['t_dat'] > history['t_dat'].max() - relativedelta(days=14)) & 
                                   (history['t_dat'] <= history['t_dat'].max() - relativedelta(days=7))]
    history_2weekago_df =  history_2weekago.groupby('customer_id').agg({'t_dat':['count', 'max', 'min'],
                                            'price':['max', 'min', 'mean'],
                                            'Ladieswear':'sum',
                                            'Divided':'sum',
                                            'Menswear':'sum',
                                            'Sport':'sum',
                                            'Baby/Children':'sum'}).reset_index() 
    history_2weekago_df.columns = ['customer_id','customer_total_cnt_2weekago', 'customer_total_latest_buy_2weekago', 'customer_total_1st_buy_2weekago', 
                                'customer_price_max_2weekago', 'customer_price_min_2weekago', 'customer_price_mean_2weekago',
                                'Ladieswear_2weekago', 'Divided_2weekago', 'Menswear_2weekago', 'Sport_2weekago', 'Baby/Children_2weekago']

    history_2weekago_df['customer_total_1st_buy_2weekago'] = (history_2weekago['t_dat'].max() - history_2weekago_df['customer_total_1st_buy_2weekago']).dt.days
    history_2weekago_df['customer_total_latest_buy_2weekago'] = (history_2weekago['t_dat'].max() - history_2weekago_df['customer_total_latest_buy_2weekago']).dt.days

    for g in group:
        history_2weekago_df[g+'_2weekago'] = history_2weekago_df[g+'_2weekago'] / history_2weekago_df['customer_total_cnt_2weekago']

    df = pd.merge(df,history_2weekago_df,how='left',on='customer_id')

    del history_2weekago, history_2weekago_df
    gc.collect()

    return df

@noglobal
def make_customer_article_features(target, history):
    df = target.merge(history, on=['customer_id', 'article_id'], how='inner')
    df = df.groupby(['customer_id', 'article_id']).agg({'t_dat':['count', 'min', 'max']}).reset_index()
    df.columns = ['customer_id', 'article_id', 'count', '1st_buy_date_diff', 'latest_buy_date_diff']
    df['1st_buy_date_diff'] = (history['t_dat'].max() - df['1st_buy_date_diff']).dt.days
    df['latest_buy_date_diff'] = (history['t_dat'].max() - df['latest_buy_date_diff']).dt.days
    return df

@noglobal
def add_same_article_type_rate(target, history, col):
    add_data = history[['customer_id', col]].copy()
    add_data['total'] = add_data.groupby('customer_id').transform('count')
    add_data = add_data.groupby(['customer_id', col])['total'].agg(['max', 'count']).reset_index()
    add_data[f'{col}_customer_buy_rate'] = add_data['count'] / add_data['max']
    target = target.merge(add_data[['customer_id', col, f'{col}_customer_buy_rate']], on=['customer_id', col], how='left')
    return target

@noglobal
def make_new_article_features(first_week_sales_pred):
    first_week_sales_pred['new_article'] = 1
    return first_week_sales_pred[['article_id', 'new_article', '1st_week_sales_pred']]
    

@noglobal
def add_features(df, history, articles, customers):
    df = df.merge(make_article_features(articles), on=['article_id'], how='left')
    df = df.merge(make_article_tran_features(history), on=['article_id'], how='left')
    df = df.merge(make_customer_features(customers), on=['customer_id'], how='left')
    df = df.merge(make_customer_tran_features(history), on=['customer_id'], how='left')
    df = df.merge(make_customer_article_features(df[['customer_id', 'article_id']], history), on=['article_id', 'customer_id'], how='left')
    # df = df.merge(make_new_article_features(first_week_sales_pred), on=['article_id'], how='left')
    # df = df.merge(text_svd_df, on=['article_id'], how='left')

    cols = ['product_type_name', 'product_group_name', 'graphical_appearance_name',
            'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name',
            'index_name', 'index_group_name', 'section_name', 'garment_group_name']

    for c in cols:
        df = add_same_article_type_rate(df, history, c)

    return df
    


@noglobal
def add_labels(recom_result, history):
    """レコメンドしたデータが学習期間で購入されたかどうかのフラグを付与する

    Args:
        recom_result (_type_): レコメンド結果
        train_tran (_type_): 学習期間のトランザクションデータ

    Returns:
        _type_: 学習期間での購入フラグを付与したレコメンド結果
    """
    history = history[['customer_id', 'article_id']].drop_duplicates()
    history['buy'] = 1
    recom_result = recom_result.merge(history, on=['customer_id', 'article_id'], how='left')
    recom_result['buy'] = recom_result['buy'].fillna(0)
    return recom_result

# copy from t88


if __name__ == '__main__':
    args = sys.argv

    offline = True
    test = False

    if False:
        INPUT_DIR = '../../h-and-m-personalized-fashion-recommendations/'
        articles = pd.read_csv(INPUT_DIR + 'articles.csv', dtype='object')
        customers = pd.read_csv(INPUT_DIR + 'customers.csv')
        transactions = pd.read_csv(INPUT_DIR + 'transactions_train.csv', dtype={'article_id':'str'}, parse_dates=['t_dat'])
        sample = pd.read_csv(INPUT_DIR + 'sample_submission.csv')

        ALL_CUSTOMER = customers['customer_id'].unique().tolist()
        ALL_ARTICLE = articles['article_id'].unique().tolist()

        customer_ids = dict(list(enumerate(ALL_CUSTOMER)))
        article_ids = dict(list(enumerate(ALL_ARTICLE)))

        customer_map = {u: uidx for uidx, u in customer_ids.items()}
        article_map = {i: iidx for iidx, i in article_ids.items()}

        with open("dataset/customer_ids.pickle", 'wb') as handle:
            pickle.dump(customer_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("dataset/article_ids.pickle", 'wb') as handle:
            pickle.dump(article_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        articles['article_id'] = articles['article_id'].map(article_map)
        customers['customer_id'] = customers['customer_id'].map(customer_map)
        transactions['article_id'] = transactions['article_id'].map(article_map)
        transactions['customer_id'] = transactions['customer_id'].map(customer_map)
        sample['customer_id'] = sample['customer_id'].map(customer_map)
        
        articles['article_id'] = articles['article_id'].astype(np.int32)
        customers['customer_id'] = customers['customer_id'].astype(np.int32)
        transactions['customer_id'] = transactions['customer_id'].astype(np.int32)
        transactions['article_id'] = transactions['article_id'].astype(np.int32)

        # 名寄せ
        customers['fashion_news_frequency'] = customers['fashion_news_frequency'].str.replace('None','NONE')
        customers['age10'] = str((customers['age'] // 10) * 10)
        customers.loc[customers['age'].isnull(), 'age10'] = np.nan

        # label_encoding
        le_cols = ['product_type_name', 'product_group_name', 'graphical_appearance_name',
                    'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name',
                    'index_name', 'index_group_name', 'section_name', 'garment_group_name']
        for c in le_cols:
            le = LabelEncoder()
            articles[c] = le.fit_transform(articles[c].fillna(''))


        le_cols = ['club_member_status', 'fashion_news_frequency', 'postal_code', 'age10']
        for c in le_cols:
            le = LabelEncoder()
            customers[c] = le.fit_transform(customers[c].fillna(''))

        customers['customer_type'] = customers['FN'].fillna(0).astype(int).astype(str) + \
                                    customers['Active'].fillna(0).astype(int).astype(str) + \
                                    customers['club_member_status'].fillna(0).astype(int).astype(str) + \
                                    customers['fashion_news_frequency'].fillna(0).astype(int).astype(str) + \
                                    customers['age10'].fillna(0).astype(int).astype(str)

        le = LabelEncoder()
        customers['customer_type'] = le.fit_transform(customers['customer_type'])

        # transactionに紐づけ
        transactions = transactions.merge(customers, on='customer_id', how='left')
        transactions = transactions.merge(articles, on='article_id', how='left')

        transactions.to_parquet('dataset/transactions_train.parquet')
        customers.to_parquet('dataset/customers.parquet')
        articles.to_parquet('dataset/articles.parquet')


    Ns = {}
    Ns['cf_a'] = 20 
    Ns['ctf_a'] = 20
    Ns['atfd_a'] = 20
    Ns['atfp_a'] = 20
    Ns['pa_a'] = 20

    Ns['cf_w'] = 20
    Ns['ctf_w'] = 20
    Ns['atfd_w'] = 20
    Ns['atfp_w'] = 20
    Ns['pa_w'] = 20

    Ns['cf_m'] = 20
    Ns['ctf_m'] = 20
    Ns['atfd_m'] = 20
    Ns['atfp_m'] = 20
    Ns['pa_m'] = 20

    Ns['cf_y'] = 20
    Ns['ctf_y'] = 20
    Ns['atfd_y'] = 20
    Ns['atfp_y'] = 20
    Ns['pa_y'] = 20
    RUN_US = True

    transactions = pd.read_parquet('dataset/transactions_train.parquet')
    customers = pd.read_parquet('dataset/customers.parquet')
    articles = pd.read_parquet('dataset/articles.parquet')


    if False:
        train_start = '2020-09-09'
        valid_start = '2020-09-16'
        valid_end = '2020-09-22'

        hist_st = train_start
        target_st = valid_start
        ml_train = pd.DataFrame()
        N_ITER = 1

        for i in range(N_ITER):
            print(i)
            history_tran = transactions[transactions['t_dat'] < valid_start].copy()
            target_tran = transactions[(transactions['t_dat'] >= valid_start) & (transactions['t_dat'] < valid_end)].copy()
            # first_week_sales_pred_tmp = first_week_sales_pred[(first_week_sales_pred['1st_week_sales_dat'] >= target_tran['t_dat'].min())&(first_week_sales_pred['1st_week_sales_dat'] <= target_tran['t_dat'].max())]
            first_week_sales_pred_tmp = None
            target_id = target_tran['customer_id'].unique().tolist()
            recom = get_reccomend(target_id, history_tran, Ns, first_week_sales_pred_tmp)
            recom.to_parquet('result/recall_t88.parquet')
            ml_train_tmp = add_labels(recom, target_tran)

            # # under sampling
            # if RUN_US:
            #     sample_n = int(ml_train_tmp['buy'].sum())
            #     ml_train_tmp0 = ml_train_tmp[ml_train_tmp['buy']==0.0].sample(sample_n*10, random_state=42)
            #     ml_train_tmp1 = ml_train_tmp[ml_train_tmp['buy']==1.0].copy()
            #     ml_train_tmp = pd.concat([ml_train_tmp0, ml_train_tmp1])

            ml_train_tmp = add_features(ml_train_tmp, history_tran, articles, customers)

            ml_train = ml_train.append(ml_train_tmp)
            ml_train['customer_id'] = ml_train['customer_id'].astype(np.int32)
            ml_train['article_id'] = ml_train['article_id'].astype(np.int32)

            ml_train.to_parquet('result/rank_train.parquet', index=False)

            # history_tran = transactions[transactions['t_dat'] < valid_start].copy()
            # target_tran = transactions[transactions['t_dat'] >= valid_start].copy()
            # # first_week_sales_pred_tmp = first_week_sales_pred[(first_week_sales_pred['1st_week_sales_dat'] >= target_tran['t_dat'].min())&(first_week_sales_pred['1st_week_sales_dat'] <= target_tran['t_dat'].max())]

            # target_id = target_tran['customer_id'].unique().tolist()
            # recom = get_reccomend(target_id, history_tran, Ns, first_week_sales_pred_tmp)
            # ml_valid = add_labels(recom, target_tran)
            # ml_valid = add_features(ml_valid, history_tran, articles, customers, first_week_sales_pred_tmp)
            # ml_valid.to_csv('result/rank_val.parquet', index=False)

if True:


    # create predcit file
    all_target_id = customers['customer_id'].tolist()
    # first_week_sales_pred_tmp = first_week_sales_pred[first_week_sales_pred['1st_week_sales_dat'] >= '2020/09/23']
    BATCH_SIZE = int(7e4)

    # メモリのケアのためバッチで推論を回す
    batchs = [all_target_id[i:i+BATCH_SIZE] for i in range(0, len(all_target_id), BATCH_SIZE)]

    if True:
        i = 0
        for target_id in tqdm(batchs):
            first_week_sales_pred_tmp = None
            recom = get_reccomend(target_id, transactions, Ns, first_week_sales_pred_tmp)
            ml_test = add_features(recom, transactions, articles, customers)
            ml_test = add_labels(ml_test, transactions)
            ml_test = ml_test.rename(columns={"buy": "label"}, errors="raise")
            ml_test.to_parquet(f'temp/for_predcit_{i}.parquet')
            i+=1

# predict
if False:


    for fold in range(5):
        print(f'fold{fold}')
        preds = []

        for file_no in tqdm(range(14)):
            ml_test = pd.read_parquet(f'temp/for_predcit_{file_no}.parquet')

            # ml_test = ml_test.rename(columns={"buy": "label"}, errors="raise")
            score_df = ml_test[['customer_id', 'article_id', 'label']]

            features = list(
            filter(lambda x: x not in ['sales_channel_id', 'customer_id', 'article_id', 'week','label'],
                ml_test.columns))

            test_pred = np.zeros(len(ml_test))
            #with open(f'model/lgb_classification{fold}.pkl', 'rb') as f:
            with open(f'model/lgb{fold}.pkl', 'rb') as f:
                model = pickle.load(f)
            score_df['pred_score'] = model.predict(ml_test[features], num_iteration=model.best_iteration_)
            
            preds.append(score_df)
        
        del  ml_test, test_pred
        gc.collect()

        test = pd.concat(preds)
        test.to_parquet(f'exp01/sub{fold}.parquet')

    # with open("dataset/customer_ids.pickle", 'rb') as file:
    #     customer_map = pickle.load(file)

    # with open("dataset/article_ids.pickle", 'rb') as file:
    #     article_map = pickle.load(file)
# ml_test = pd.read_parquet(f'temp/for_predcit_0.parquet')
# # ml_test = ml_test.rename(columns={"buy": "label"}, errors="raise")
# print(ml_test.dtypes)