import os
import random
import time
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
import pathlib
import cudf
import numpy as np
from utils import Logger, evaluate, reduce_mem
import signal
import multitasking
import gc
import datetime

seed = 42
random.seed(seed)

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

# init log
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('rank lgb')


def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(
            lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


def train_lgb(df_train_set, topk=12, offline=True):
    ts = time.time()

    """
    :param df_train_set
    :return:
    """
    fold = 2
    ycol = 'label'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'sales_channel_id', 'customer_id', 'article_id'], df_train_set.columns))

    df_importance_list = []
    score_list = []
    score_df = df_train_set[['customer_id', 'article_id', 'label']]

    kfold = GroupKFold(n_splits=fold)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train_set[feature_names], df_train_set[ycol], df_train_set['customer_id'])):

        X_train = df_train_set.iloc[trn_idx]
        Y_train = df_train_set.iloc[trn_idx]

        X_val = df_train_set.iloc[val_idx][ycol]
        Y_val = df_train_set.iloc[val_idx][ycol]


        lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                        max_depth=-1, n_estimators=1000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                        learning_rate=0.01, min_child_weight=50, random_state=42, n_jobs=-1)

        lgb_Classfication.fit(X_train[feature_names], X_val, eval_set=[(Y_train[feature_names], Y_val)], eval_metric=['auc', ], early_stopping_rounds=50, verbose=100)

        joblib.dump(lgb_Classfication, f'model/lgb{fold_id}.pkl')

        df_importance = pd.DataFrame({
            'feature_name':
                feature_names,
                'importance':
                lgb_Classfication.feature_importances_,
        })

        df_importance_list.append(df_importance)

        Y_train['pred_score'] = lgb_Classfication.predict_proba(
            Y_train[feature_names], num_iteration=lgb_Classfication.best_iteration_)[:,1]

        Y_train.sort_values(by=['customer_id', 'pred_score'])
        Y_train['pred_rank'] = Y_train.groupby(
            ['customer_id'])['pred_score'].rank(ascending=False, method='first')

        score_list.append(
            Y_train[['customer_id', 'article_id', 'pred_score', 'pred_rank']])

        del X_train, Y_train, X_val, Y_val, g_train, g_eval
        gc.collect()

    del df_train_set
    gc.collect()

    log.debug('*'*20)
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')
    log.debug('*'*20)
    score_df_ = pd.concat(score_list, axis=0)
    score_df = score_df.merge(score_df_, how='left', on=[
                              'customer_id', 'article_id'])

    score_df[['customer_id', 'article_id', 'pred_score', 'pred_rank',
              'label']].to_csv('result/trn_lgb_classification_feats.csv', index=False)


def train_lgb_rank(df_train_set, topk=12, offline=True):
    ts = time.time()

    """
    :param df_train_set
    :return:
    """
    fold = 2
    ycol = 'label'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'sales_channel_id', 'customer_id', 'article_id'], df_train_set.columns))

    df_importance_list = []
    score_list = []
    score_df = df_train_set[['customer_id', 'article_id', 'label']]

    kfold = GroupKFold(n_splits=fold)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train_set[feature_names], df_train_set[ycol], df_train_set['customer_id'])):

        X_train = df_train_set.iloc[trn_idx]
        Y_train = df_train_set.iloc[trn_idx]

        X_val = df_train_set.iloc[val_idx][ycol]
        Y_val = df_train_set.iloc[val_idx][ycol]

        g_train = X_train.groupby(['customer_id'], as_index=False).count()[
            ycol].values
        g_eval = Y_train.groupby(['customer_id'], as_index=False).count()[
            ycol].values

        lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                    max_depth=-1, n_estimators=1000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                    learning_rate=0.01, min_child_weight=50, random_state=42, n_jobs=-1)

        lgb_ranker.fit(X_train[feature_names], X_val, group=g_train, eval_set=[(Y_train[feature_names], Y_val)], eval_group=[
                       g_eval], early_stopping_rounds=50, verbose=100, eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ])

        joblib.dump(lgb_ranker, f'model/lgb{fold_id}.pkl')

        df_importance = pd.DataFrame({
            'feature_name':
                feature_names,
                'importance':
                lgb_ranker.feature_importances_,
        })

        df_importance_list.append(df_importance)

        Y_train['pred_score'] = lgb_ranker.predict(
            Y_train[feature_names], num_iteration=lgb_ranker.best_iteration_)
        Y_train['pred_score'] = Y_train[['pred_score']
                                        ].transform(lambda x: norm_sim(x))

        Y_train.sort_values(by=['customer_id', 'pred_score'])
        Y_train['pred_rank'] = Y_train.groupby(
            ['customer_id'])['pred_score'].rank(ascending=False, method='first')

        score_list.append(
            Y_train[['customer_id', 'article_id', 'pred_score', 'pred_rank']])

        del X_train, Y_train, X_val, Y_val, g_train, g_eval
        gc.collect()

    del df_train_set
    gc.collect()

    log.debug('*'*20)
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')
    log.debug('*'*20)
    score_df_ = pd.concat(score_list, axis=0)
    score_df = score_df.merge(score_df_, how='left', on=[
                              'customer_id', 'article_id'])
    score_df[['customer_id', 'article_id', 'pred_score', 'pred_rank',
              'label']].to_csv('result/trn_lgb_ranker_feats.csv', index=False)

def submit(recall_df, topk=12, model_name='formmat'):

    INPUT_DIR = '/home/shi/workspace/h-and-m-personalized-fashion-recommendations/'
    articles = pd.read_csv(INPUT_DIR + 'articles.csv', dtype={"article_id": "str"})[['article_id']]
    customers = pd.read_csv(INPUT_DIR + 'customers.csv')[['customer_id']]
    ALL_CUSTOMER = customers['customer_id'].unique().tolist()
    ALL_ARTICLE = articles['article_id'].unique().tolist()
    customer_ids = dict(list(enumerate(ALL_CUSTOMER)))
    article_ids = dict(list(enumerate(ALL_ARTICLE)))
    customer_map = {u: uidx for uidx, u in customer_ids.items()}
    article_map = {i: iidx for iidx, i in article_ids.items()}
    customers['customer_ids_int'] = customers['customer_id'].map(customer_map)
    articles['article_ids_int'] = articles['article_id'].map(article_map)

    recall_df = recall_df.sort_values(by=['customer_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['customer_id'])['pred_score'].rank(ascending=False, method='first')
    
    tmp = recall_df.groupby('customer_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk]
    submit = submit.rename(columns={"customer_id": "customer_ids_int", "article_id": "article_ids_int"})
    submit = submit.merge(articles, how='left')
    submit = submit.merge(customers, how='left')[['customer_id', 'article_id']]

    submit = submit.groupby('customer_id')['article_id'].apply(list).reset_index()
    submit = submit.rename(columns={'article_id':'prediction'})
    submit['prediction'] = submit['prediction'].apply(lambda x: ' '.join(x))
    
    save_name = model_name + '_' + datetime.datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)

def _stacking():
    trn_lgb_ranker_feats = pd.read_csv('result/trn_lgb_ranker_feats.csv')
    trn_lgb_cls_feats = pd.read_csv('result/trn_lgb_cls_feats.csv')
    pass

if __name__ == '__main__':

    df = pd.read_csv('result/trn_lgb_ranker_feats.csv')
    df['article_id'] = df['article_id'].astype(np.int32)
    submit(df)
    # print(df.head())
    # target_ids = pd.read_csv(INPUT_DIR + 'sample_submission.csv')['customer_id'].tolist()
    # predict(target_ids)
