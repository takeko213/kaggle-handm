import os
import random
import time
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cudf
import numpy as np
from utils_log import Logger
from utils_evaluate import evaluate

seed = 42
random.seed(seed)

# init log
log_file = 'my_log.log'
os.makedirs('log', exist_ok=True)
log = Logger(f'log/{log_file}').logger
log.info('rank lgb')


def train_model(df_train_set, label):
    """
    :param df_train_set
    :return:
    """
    fold=2
    ycol = 'label'
    feature_names = list(
        filter(lambda x: x not in [ycol, 't_dat', 'sales_channel_id','customer_id', 'article_id'], df_train_set.columns))
    print(feature_names)
    feature_names.sort()

    model = lgb.LGBMClassifier(num_leaves=64,
                               max_depth=10,
                               learning_rate=0.05,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.8,
                               reg_alpha=0.5,
                               reg_lambda=0.5,
                               random_state=seed,
                               importance_type='gain',
                               metric=None)

    oof = []
    prediction = df_train_set[['customer_id', 'article_id']]
    prediction['pred'] = 0
    df_importance_list = []

    # train
    kfold = GroupKFold(n_splits=fold)
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train_set[feature_names], df_train_set[ycol], df_train_set['customer_id'])):
        X_train = df_train_set.iloc[trn_idx][feature_names]
        Y_train = df_train_set.iloc[trn_idx][ycol]

        X_val = df_train_set.iloc[val_idx][feature_names]
        Y_val = df_train_set.iloc[val_idx][ycol]

        log.debug(
            f'\nFold_{fold_id + 1} Training ================================\n'
        )

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=100,
                              eval_metric='auc',
                              early_stopping_rounds=100)

        pred_val = lgb_model.predict_proba(X_val, num_iteration=lgb_model.best_iteration_)[:, 1]
        df_oof = df_train_set.iloc[val_idx][['customer_id', 'article_id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = lgb_model.predict_proba(
            df_train_set[feature_names], num_iteration=lgb_model.best_iteration_)[:, 1]
        prediction['pred'] += pred_test / 2

        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        joblib.dump(model, f'model/lgb{fold_id}.pkl')

    
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')


    df_oof = pd.concat(oof)
    df_oof.sort_values(['customer_id', 'pred'],
                       inplace=True,
                       ascending=[True, False])
    log.debug(f'df_oof.head: {df_oof.head()}')
 

    total = df_train_set.customer_id.nunique()

    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_oof, total)

    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )




# def train_and_predict(df_train_set, topk=12, offline=True):
#     ts = time.time()
#     seed = 42
#     ycol = 'label'
#     feature_names = list(
#         filter(lambda x: x not in [ycol, 't_dat', 'sales_channel_id','customer_id', 'article_id'], df_train_set.columns))

#     feature_names.sort()

#     train = df_train_set.copy()
#     X = train.copy()
#     y = train['label']

    # check user count
    # print(df_train_set.columns)
    # test = train.groupby('customer_id')['label'].agg('count').reset_index()
    # test.columns = ['customer_id', 'cnt']
    # # test = test.sort_values(['customer_id'],ascending=True)
    # print(len(test))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    # X_eval, X_off, y_eval, y_off = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
    # g_train = X_train.groupby(['customer_id'], as_index=False).count()['label'].values
    # g_eval = X_eval.groupby(['customer_id'], as_index=False).count()['label'].values
    # lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    #                             max_depth=-1, n_estimators=1000, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    #                             learning_rate=0.01, min_child_weight=50, random_state=66, n_jobs=-1)

    # lgb_ranker.fit(X_train[feature_names], y_train, group=g_train, eval_set=[(X_eval[feature_names], y_eval)], eval_group=[g_eval], early_stopping_rounds=50, verbose=False)
    
    # # joblib.dump(model, f'model/lgb{fold_id}.pkl')


    # def print_feature_importance(lgb_ranker):
    #     df_importance_list = []
    #     df_importance = pd.DataFrame({
    #             'feature_name':
    #             feature_names,
    #             'importance':
    #             lgb_ranker.feature_importances_,
    #         })
    #     df_importance_list.append(df_importance)
    #     log.debug('*'*20)
    #     df_importance = pd.concat(df_importance_list)
    #     df_importance = df_importance.groupby([
    #         'feature_name'
    #     ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    #     log.debug(f'importance: {df_importance}')
    #     log.debug('*'*20)
    
    # print_feature_importance(lgb_ranker)
    
    # X_off['pred_score'] = lgb_ranker.predict(X_off[feature_names], num_iteration=lgb_ranker.best_iteration_)
    # recall_df = X_off[['customer_id', 'article_id', 'pred_score']].copy()
    # recall_df = recall_df.sort_values(by=['customer_id', 'pred_score'])
    # recall_df['rank'] = recall_df.groupby(['customer_id'])['pred_score'].rank(ascending=False, method='first')

    # check top counts 
    # tmp = recall_df.groupby('customer_id').apply(lambda x: x['rank'].max())
    # assert tmp.min() >= topk
    # todo
#     del recall_df['pred_score'], recall_df['label']
#     submit = recall_df[recall_df['rank'] <= 5].set_index(['user_id', 'rank']).unstack(-1).reset_index()
#     max_article = int(recall_df['rank'].value_counts().index.max())
#     submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]


if __name__ == '__main__':

    df_train_feature = pd.read_csv('data/rank_train.csv')
    print(len(df_train_feature['customer_id'].unique()))

    INPUT_DIR = 'dataset/'
    transactions = cudf.read_parquet(INPUT_DIR + 'transactions.parquet')
    transactions.t_dat = cudf.to_datetime(transactions.t_dat)
    label = transactions[transactions.t_dat > np.datetime64('2020-09-16')][['customer_id', 'article_id']]
    label['label2'] = 1
    print(label.head())
    train_model(df_train_feature, label)
