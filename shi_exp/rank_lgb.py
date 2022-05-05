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
from xgboost import train
from utils import Logger, reduce_mem, metrics
import signal
import multitasking
import gc
import datetime
import sys
import pickle

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
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(
            lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['customer_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


def train_lgb_rank(df_train):
    ts = time.time()

    """
    :param df_train
    :return:
    """

    fold = 5
    ycol = 'label'
    df_train = df_train.rename(columns={"buy": "label"}, errors="raise")
    feature_names = list(
        filter(lambda x: x not in [ycol, 'sales_channel_id', 'customer_id', 'article_id', 'week'],
               df_train.columns))
    print(feature_names)
    df_importance_list = []
    score_list = []
    
    score_df = df_train[['customer_id', 'article_id', 'label']]

    user_set = get_kfold_users(df_train, n=fold)

    for fold_id, valid_user in enumerate(user_set):
        X_train = df_train[~df_train['customer_id'].isin(valid_user)]
        Y_train = df_train[df_train['customer_id'].isin(valid_user)]

        g_train = X_train.groupby(['customer_id'], as_index=False).count()[
            ycol].values
        g_eval = Y_train.groupby(['customer_id'], as_index=False).count()[
            ycol].values

        lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                    max_depth=-1, n_estimators=5000, subsample=0.7, colsample_bytree=0.7,
                                    subsample_freq=1,
                                    learning_rate=0.01, min_child_weight=50, random_state=42, n_jobs=-1)

        # lgb_ranker = lgb.LGBMRanker(
        #     objective="lambdarank",
        #     metric="ndcg",
        #     boosting_type="dart",
        #     n_estimators=1,
        #     importance_type='gain',
        #     force_col_wise=True,
        #     verbose=10
        # )

        lgb_ranker = lgb_ranker.fit(
            X_train[feature_names],
            X_train[ycol],
            group=g_train,
        )

        # lgb_ranker.fit(X_train[feature_names], X_train[ycol], group=g_train,
        #                eval_set=[(Y_train[feature_names], Y_train[ycol])], eval_group=[
        #         g_eval], early_stopping_rounds=200, verbose=100, eval_at=[12], eval_metric=['ndcg', ])

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

        del X_train, Y_train, g_train, g_eval
        gc.collect()

    del df_train
    gc.collect()

    log.debug('*' * 20)
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')
    log.debug('*' * 20)
    score_df_ = pd.concat(score_list, axis=0)
    score_df = score_df.merge(score_df_, how='left', on=[
        'customer_id', 'article_id'])

    score_df[['customer_id', 'article_id', 'pred_score', 'pred_rank',
              'label']].to_parquet('result/trn_lgb_ranker_feats.parquet', index=False)


def train_lgb(df_train):

    params = {
        "objective": "binary",
        "boosting": "gbdt",
        "learning_rate": 0.01,
        "metric": "binary_logloss",
        "seed": 42,
        # 'verbose':-1
    }

    ycol = 'label'
    df_train = df_train.rename(columns={"buy": "label"}, errors="raise")
    fi = pd.DataFrame()
    feature_names = list(
        filter(lambda x: x not in [ycol, 'sales_channel_id', 'customer_id', 'article_id', 'week'],
               df_train.columns))
    fold = 5

    df_importance_list = []
    score_list = []
 
    score_df = df_train[['customer_id', 'article_id', 'label']]
    user_set = get_kfold_users(df_train, n=fold)

    for fold_id, valid_user in enumerate(user_set):
        X_train = df_train[~df_train['customer_id'].isin(valid_user)]
        Y_train = df_train[df_train['customer_id'].isin(valid_user)]

        tr_data = lgb.Dataset(X_train[feature_names], label=X_train[ycol])
        vl_data = lgb.Dataset(Y_train[feature_names], label=Y_train[ycol])

        model = lgb.train(params, tr_data, valid_sets=[tr_data, vl_data],
                          num_boost_round=20000, early_stopping_rounds=100, verbose_eval=1000)

        joblib.dump(model, f'model/lgb_classification{fold_id}.pkl')

        df_importance = pd.DataFrame({
            'feature_name':
                feature_names,
            'importance':
                model.feature_importance,
        })

        df_importance_list.append(df_importance)

        Y_train['pred_score'] = model.predict(
            Y_train[feature_names], num_iteration=model.best_iteration)

        # Y_train['pred_score'] = Y_train[['pred_score']
        #                                 ].transform(lambda x: norm_sim(x))

        Y_train.sort_values(by=['customer_id', 'pred_score'])
        Y_train['pred_rank'] = Y_train.groupby(
            ['customer_id'])['pred_score'].rank(ascending=False, method='first')

        score_list.append(
            Y_train[['customer_id', 'article_id', 'pred_score', 'pred_rank']])

        # del X_train, Y_train, g_train, g_eval
        gc.collect()

    del df_train
    gc.collect()

    # log.debug('*' * 20)
    # df_importance = pd.concat(df_importance_list)
    # df_importance = df_importance.groupby([
    #     'feature_name'
    # ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    # log.debug(f'importance: {df_importance}')
    # log.debug('*' * 20)
    # score_df_ = pd.concat(score_list, axis=0)
    # score_df = score_df.merge(score_df_, how='left', on=[
    #     'customer_id', 'article_id'])
    score_df_ = pd.concat(score_list, axis=0)
    score_df = score_df.merge(score_df_, how='left', on=[
        'customer_id', 'article_id'])

    score_df[['customer_id', 'article_id', 'pred_score', 'pred_rank',
              'label']].to_parquet('result/trn_lgb_classifer_feats.parquet', index=False)
    # fi_tmp = pd.DataFrame()
    # fi_tmp['iter'] = 1
    # fi_tmp['feature'] = model.feature_name()
    # fi_tmp['importance'] = model.feature_importance(importance_type='gain')
    # fi = fi.append(fi_tmp)

    # # # cv
    # vl_pred = model.predict(vl_x, num_iteration=model.best_iteration)
    # valid_pred = df_val[['customer_id', 'article_id']].copy()
    # valid_pred['pred_score'] = vl_pred
    # valid_pred[['customer_id', 'article_id', 'pred_score']].to_parquet('result/trn_lgb_feats.parquet', index=False)


def calculate_cv(df_test, df_predict):
    '''
        calculate cv
    '''
    df_test = df_test.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    df_test.columns = ['customer_id', 'valid_true']
    # df_predict = pd.read_parquet('result/trn_lgb_ranker_feats.parquet')
    df_predict = df_predict.sort_values(
        ['customer_id', 'pred_score'], ascending=False)
    df_predict = df_predict.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    df_predict.columns = ['customer_id', 'valid_pred']

    merge = df_test.merge(df_predict, how='inner')
    metrics(merge, 12)

# def _stacking():
#     trn_lgb_ranker_feats = pd.read_csv('result/trn_lgb_ranker_feats.csv')
#     trn_lgb_cls_feats = pd.read_csv('result/trn_lgb_cls_feats.csv')
#     pass



# def submmit(sample):

#     with open("dataset/customer_ids.pickle", 'rb') as file:
#         customer_map = pickle.load(file)

#     with open("dataset/article_ids.pickle", 'rb') as file:
#         article_map = pickle.load(file)

#     all_target_id = sample['customer_id'].tolist()
#     # first_week_sales_pred_tmp = first_week_sales_pred[first_week_sales_pred['1st_week_sales_dat'] >= '2020/09/23']

#     # メモリのケアのためバッチで推論を回す
#     batchs = [all_target_id[i:i+BATCH_SIZE] for i in range(0, len(all_target_id), BATCH_SIZE)]
    
#     preds = []
    
#     for target_id in batchs:
#         recom = get_reccomend(target_id, transactions, Ns, first_week_sales_pred_tmp)
#         ml_test = add_features(recom, transactions, articles, customers, first_week_sales_pred_tmp, text_svd_df)

#         test_pred = np.zeros(len(ml_test))
#         models = pathlib.Path(OUTPUT_DIR + f'{exp_name}').glob('model*.pickle')

#         for m in models:
#             with open(m, 'rb') as f:
#                 model = pickle.load(f)
#         test_pred += model.predict(ml_test[features], num_iteration=model.best_iteration) / N_SEED

#         test = ml_test[['customer_id', 'article_id']].copy()
#         test['prob'] = test_pred
#         test = test.sort_values(['customer_id', 'prob'], ascending=False)
#         test = test.groupby('customer_id').head(12)
#         preds.append(test)
    
#     del recom, ml_test, test_pred
#     gc.collect()

#     test = pd.concat(preds)
#     test['article_id'] = test['article_id'].map(article_ids)
#     test['customer_id'] = test['customer_id'].map(customer_ids)

#     test = test.groupby('customer_id')['article_id'].apply(list).reset_index()

#     sub = sample['customer_id'].map(customer_ids).to_frame()
#     sub = sub.merge(test, on=['customer_id'], how='left')
#     sub = sub.rename(columns={'article_id':'prediction'})
#     assert(sub['prediction'].apply(len).min()==12)
#     sub['prediction'] = sub['prediction'].apply(lambda x: ' '.join(x))
#     sub.to_csv(OUTPUT_DIR + f'{exp_name}/{exp_name}_sub.csv', index=False)
    
def submit(recall_df, topk=12, model_name='formmat'):
    INPUT_DIR = '/home/shi/workspace/h-and-m-personalized-fashion-recommendations/'
    articles = pd.read_csv(INPUT_DIR + 'articles.csv',
                           dtype={"article_id": "str"})[['article_id']]
    customers = pd.read_csv(INPUT_DIR + 'customers.csv')[['customer_id']]

    with open("dataset/customer_ids.pickle", 'rb') as file:
        customer_map = pickle.load(file)

    with open("dataset/article_ids.pickle", 'rb') as file:
        article_map = pickle.load(file)

    customers['customer_ids_int'] = customers['customer_id'].map(customer_map)
    articles['article_ids_int'] = articles['article_id'].map(article_map)

    recall_df['pred_score'] = recall_df[['pred_score']
                                        ].transform(lambda x: norm_sim(x))

    recall_df = recall_df.sort_values(by=['customer_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(
        ['customer_id'])['pred_score'].rank(ascending=False, method='first')

    tmp = recall_df.groupby('customer_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk]
    submit = submit.rename(
        columns={"customer_id": "customer_ids_int", "article_id": "article_ids_int"})
    submit = submit.merge(articles, how='left')
    submit = submit.merge(customers, how='left')[['customer_id', 'article_id']]

    submit = submit.groupby('customer_id')[
        'article_id'].apply(list).reset_index()
    submit = submit.rename(columns={'article_id': 'prediction'})
    submit['prediction'] = submit['prediction'].apply(lambda x: ' '.join(x))

    save_name = model_name + '_' + datetime.datetime.today().strftime('%m-%d-%H-%M') + '.csv'
    submit.to_csv(save_name, index=False, header=True)

if __name__ == '__main__':
        
    if True:
        # infer
        with open("dataset/customer_ids.pickle", 'rb') as file:
            customer_map = pickle.load(file)

        with open("dataset/article_ids.pickle", 'rb') as file:
            article_map = pickle.load(file)
            
        
        pred_score = []
        for i in range(5):
            df = pd.read_parquet(f'exp01/sub{i}.parquet')
            pred_score.append(df['pred_score'].values)
            

        df['article_id'] = df['article_id'].astype(np.int32)
        df['pred_score'] = np.mean(np.stack(pred_score), axis=0)

        df = df.sort_values(by=['customer_id', 'pred_score'], ascending=False)
        df['rank'] = df.groupby(
            ['customer_id'])['pred_score'].rank(ascending=False, method='first')

        df = df[df['rank'] <= 12]
        print(df.head())
        df['article_id'] = df['article_id'].map(article_map)
        df['customer_id'] = df['customer_id'].map(customer_map)

        df = df.groupby('customer_id')[
            'article_id'].apply(list).reset_index()
        df = df.rename(columns={'article_id': 'prediction'})
        df['prediction'] = df['prediction'].apply(lambda x: ' '.join(x))
        print(df.head())
        df.to_csv('result.csv', index=False)
