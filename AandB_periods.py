import pandas as pd
import numpy as np
import os
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def compute_feature(df):
    df['WKD_TYP_CD']=df['WKD_TYP_CD'].map({'WN':0,'SN': 1, 'NH': 1, 'SS': 1, 'WS': 0})
    df['date']=pd.to_datetime(df['date'])
    df['dayofweek']=df['date'].dt.dayofweek+1
    df['day']=df['date'].dt.day
    df['month']=df['date'].dt.month
    df['year']=df['date'].dt.year
    # df.drop(['date','post_id'],axis=1,inplace=True)
    season_dict = { # 冬夏长，春秋短
        1: 3, 2: 3, 3: 0,
        4: 0, 5: 1, 6: 1,
        7: 1, 8: 1, 9: 2,
        10: 2, 11: 3, 12: 3,
    }
    df['season'] = df['month'].map(season_dict)
    df.loc[df['day'] <= 10, 'tenDay'] = 0
    df.loc[(df['day'] > 10) & (df['day'] <= 20), 'tenDay'] = 1
    df.loc[df['day'] > 20, 'tenDay'] = 2
    return df
#
def lgb_cv(train_x, train_y, test_x):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    folds = 10
    seed = 2028
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    train = np.zeros((train_x.shape[0]))
    test = np.zeros((test_x.shape[0]))
    test_pre = np.zeros((folds, test_x.shape[0]))
    test_pre_all = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    tpr_scores = []
    cv_rounds = []

    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        train_matrix = lightgbm.Dataset(tr_x, label=tr_y)
        test_matrix = lightgbm.Dataset(te_x, label=te_y)
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression_l1',
            'metrics':'l1',
            # 'num_leaves': 2 ** 5-1,
            # 'feature_fraction': 0.8,
            # 'bagging_fraction': 0.8,
            'learning_rate': 0.01,
            'min_data_in_leaf': 5,
            'seed': 2028,
            'nthread': 2,
            'verbose': -1,
        }
        num_round = 4000
        early_stopping_rounds = 200
        if test_matrix:
            model = lightgbm.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=200,
                              #feval=tpr_eval_score,
                              early_stopping_rounds=early_stopping_rounds
                              )
            print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))[:10]))
            importance_list=[ x[0] for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                        key=lambda x: x[1],reverse=True))]
            #print(importance_list)
            pre = model.predict(te_x, num_iteration=model.best_iteration)#
            pred = model.predict(test_x, num_iteration=model.best_iteration)#
            train[test_index] = pre
            test_pre[i, :] = pred
            cv_scores.append(mean_squared_error (te_y, pre))
            cv_rounds.append(model.best_iteration)
            test_pre_all[i, :] = pred
        #
        print("cv_score is:", cv_scores)
    use_mean=True
    if use_mean:
        test[:] = test_pre.mean(axis=0)
    else:
        pass
    #
    print("val_mean:" , np.mean(cv_scores))
    print("val_std:", np.std(cv_scores))
    return train, test, test_pre_all, np.mean(cv_scores),importance_list
#
if __name__=="__main__":

    #
    train_df=pd.read_csv('./train_v2.csv')
    wkd_df=pd.read_csv('./wkd_v2.csv')
    wkd_df=wkd_df.rename(columns={'ORIG_DT':'date'})
    train_df=train_df.merge(wkd_df)
    #
    train_hour_df_A_cls=train_df[train_df['post_id']=='A'].reset_index(drop=True)
    train_hour_df_B=train_df[train_df['post_id']=='B'].reset_index(drop=True)
    #
    name_A=['A'+str(i) for i in range(2,14)]
    train_hour_df_A=train_hour_df_A_cls[train_hour_df_A_cls['biz_type']=='A1'].reset_index(drop=True)
    new_train_amount=train_hour_df_A['amount'].values
    for Ai in name_A:
        tmp=train_hour_df_A_cls[train_hour_df_A_cls['biz_type']==Ai].reset_index(drop=True)
        new_train_amount+=tmp['amount'].values
    #
    train_hour_df_A['amount']=new_train_amount
    train_hour_df_A.drop(['biz_type'],axis=1,inplace=True)
    train_hour_df_B.drop(['biz_type'],axis=1,inplace=True)
    train_hour_df_A=compute_feature(train_hour_df_A)
    train_hour_df_B=compute_feature(train_hour_df_B)
    #-----------
    test_df=pd.read_csv('./test_v2_periods.csv')#按0.5h计算
    test_day=pd.read_csv('./test_v2_day.csv')#按天计算
    test_hour_df_A=test_df[test_df['post_id']=='A'].reset_index(drop=True)
    test_hour_df_B=test_df[test_df['post_id']=='B'].reset_index(drop=True)
    test_hour_df_A=test_hour_df_A.merge(wkd_df)
    test_hour_df_B=test_hour_df_B.merge(wkd_df)
    test_hour_df_A=compute_feature(test_hour_df_A)
    test_hour_df_B=compute_feature(test_hour_df_B)
    test_hour_df_A.drop(['amount'],axis=1,inplace=True)
    test_hour_df_B.drop(['amount'],axis=1,inplace=True)
    #
    print(train_hour_df_A.shape,test_hour_df_A.shape)#(1035, 7) (30, 6)

    biz = ['A', 'B']
    merge_cols = ['trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                   'CJ', 'CJ_lower', 'CJ_upper', 'CW', 'CW_lower', 'CW_upper', 'CX',
                   'CX_lower', 'CX_upper', 'CY', 'CY_lower', 'CY_upper', 'DW', 'DW_lower',
                   'DW_upper', 'DZ', 'DZ_lower', 'DZ_upper', 'ET', 'ET_lower', 'ET_upper',
                   'FN', 'FN_lower', 'FN_upper', 'GQ', 'GQ_lower', 'GQ_upper', 'LB',
                   'LB_lower', 'LB_upper', 'LD', 'LD_lower', 'LD_upper', 'PAY',
                   'PAY_lower', 'PAY_upper', 'QM', 'QM_lower', 'QM_upper', 'QR',
                   'QR_lower', 'QR_upper', 'QX', 'QX_lower', 'QX_upper', 'SD', 'SD_lower',
                   'SD_upper', 'XN', 'XN_lower', 'XN_upper', 'YD', 'YD_lower', 'YD_upper',
                   'YDQ', 'YDQ_lower', 'YDQ_upper', 'YQ', 'YQ_lower', 'YQ_upper', 'YX',
                   'YX_lower', 'YX_upper', 'ZQ', 'ZQ_lower', 'ZQ_upper', 'ZY', 'ZY_lower',
                   'ZY_upper', 'additive_terms', 'additive_terms_lower',
                   'additive_terms_upper', 'holidays', 'holidays_lower', 'holidays_upper',
                   'monthly', 'monthly_lower', 'monthly_upper', 'weekly', 'weekly_lower',
                   'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper',
                   'multiplicative_terms', 'multiplicative_terms_lower',
                   'multiplicative_terms_upper', 'yhat']
    for i in range(1, 49):
        for j in biz:
            feature = pd.read_csv('./features/feature_' + j + '_' + str(i) + '.csv')
            feature = feature.rename(columns={'ds': 'date'})
            feature.drop(['Unnamed: 0'], axis=1, inplace=True)
            feature['date'] = pd.to_datetime(feature['date'])
            train_last_date = '2020-11-30'
            test_last_date = '2020-12-31'

            if j == 'A':
                train_hour_df_A.loc[train_hour_df_A['periods'] == i, merge_cols] = feature.loc[feature['date'] <= train_last_date, merge_cols].values
                test_hour_df_A.loc[train_hour_df_A['periods'] == i, merge_cols] = feature.loc[(feature['date'] > train_last_date) & (feature['date'] <= test_last_date), merge_cols].values
            else:
                train_hour_df_B.loc[train_hour_df_B['periods'] == i, merge_cols] = feature.loc[feature['date'] <= train_last_date, merge_cols].values
                test_hour_df_B.loc[test_hour_df_B['periods'] == i, merge_cols] = feature.loc[(feature['date'] > train_last_date) & (feature['date'] <= test_last_date), merge_cols].values


    train_hour_df_A = train_hour_df_A.loc[(train_hour_df_A['periods'] >= 18) & (train_hour_df_A['periods'] <= 37), :].reset_index(drop=True)
    train_hour_df_B = train_hour_df_B.loc[(train_hour_df_B['periods'] >= 18) & (train_hour_df_B['periods'] <= 37), :].reset_index(drop=True)

    select_frts=['periods','WKD_TYP_CD','year','month','day','dayofweek', 'season', 'tenDay',
                 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                   'CJ', 'CJ_lower', 'CJ_upper', 'CW', 'CW_lower', 'CW_upper', 'CX',
                   'CX_lower', 'CX_upper', 'CY', 'CY_lower', 'CY_upper', 'DW', 'DW_lower',
                   'DW_upper', 'DZ', 'DZ_lower', 'DZ_upper', 'ET', 'ET_lower', 'ET_upper',
                   'FN', 'FN_lower', 'FN_upper', 'GQ', 'GQ_lower', 'GQ_upper', 'LB',
                   'LB_lower', 'LB_upper', 'LD', 'LD_lower', 'LD_upper', 'PAY',
                   'PAY_lower', 'PAY_upper', 'QM', 'QM_lower', 'QM_upper', 'QR',
                   'QR_lower', 'QR_upper', 'QX', 'QX_lower', 'QX_upper', 'SD', 'SD_lower',
                   'SD_upper', 'XN', 'XN_lower', 'XN_upper', 'YD', 'YD_lower', 'YD_upper',
                   'YDQ', 'YDQ_lower', 'YDQ_upper', 'YQ', 'YQ_lower', 'YQ_upper', 'YX',
                   'YX_lower', 'YX_upper', 'ZQ', 'ZQ_lower', 'ZQ_upper', 'ZY', 'ZY_lower',
                   'ZY_upper', 'additive_terms', 'additive_terms_lower',
                   'additive_terms_upper', 'holidays', 'holidays_lower', 'holidays_upper',
                   'monthly', 'monthly_lower', 'monthly_upper', 'weekly', 'weekly_lower',
                   'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper',
                   'multiplicative_terms', 'multiplicative_terms_lower',
                   'multiplicative_terms_upper', 'yhat']
    cat_features = ['periods', 'WKD_TYP_CD', 'year', 'month', 'day', 'dayofweek', 'season', 'tenDay']
    
    train_hour_df_A[cat_features] = train_hour_df_A[cat_features].astype('category')
    test_hour_df_A[cat_features] = test_hour_df_A[cat_features].astype('category')
    train_hour_df_B[cat_features] = train_hour_df_B[cat_features].astype('category')
    test_hour_df_B[cat_features] =  test_hour_df_B[cat_features].astype('category')


    result = pd.read_csv('./test_v2_periods.csv')
    for i in range(18, 37):
        train_df = train_hour_df_A.loc[train_hour_df_A['periods'] == i, :].reset_index(drop=True)  # 训练集
        train_df = train_df.loc[(train_df['date'] != '2019-12-09') & (train_df['date'] != '2019-11-25'), :].reset_index(
            drop=True)
        train_df = train_df.loc[(train_df['date'] > '2020-01-05') | (train_df['date'] < '2019-12-25'), :].reset_index(
            drop=True)
        train_df = train_df.loc[train_df['amount'] != 0, :].reset_index(drop=True)
        train_df = train_df.loc[(train_df['date'] >= '2020-04-01') | (train_df['date'] <= '2020-01-28'), :].reset_index(
            drop=True)
        test_df = test_hour_df_A.loc[test_hour_df_A['periods'] == i, :].reset_index(drop=True)  # 测试集
        train_x = train_df[select_frts].copy() # (1035, 90)
        train_y = train_df['amount']
        test_x = test_df[select_frts].copy() # (30, 90)
        print(train_x.shape, train_y.shape, test_x.shape)
        lgb_train, lgb_test, sb, cv_scores, importance_list = lgb_cv(train_x, train_y, test_x) # lgb_test.shape = (30,)
        lgb_test_A = [item if item > 0 else 0 for item in lgb_test] # 取大于零的值
        result.loc[(result['post_id'] == 'A') & (result['periods'] == i), 'amount'] = lgb_test_A

    # # 对A37做特殊处理
    i = 37
    train_df = train_hour_df_A.loc[train_hour_df_A['periods'] == i, :].reset_index(drop=True)  # 训练集
    # train_df = train_df.loc[train_df['date'] >= '2020-04-25', :].reset_index(drop=True)
    train_df = train_df.loc[(train_df['date'] >= '2020-04-01') | (train_df['date'] <= '2020-01-28'), :].reset_index(
        drop=True)
    train_df = train_df.loc[(train_df['date'] != '2019-12-09') & (train_df['date'] != '2019-11-25'), :].reset_index(
        drop=True)
    train_df = train_df.loc[(train_df['date'] > '2020-01-05') | (train_df['date'] < '2019-12-25'), :].reset_index(
        drop=True)
    test_df = test_hour_df_A.loc[test_hour_df_A['periods'] == i, :].reset_index(drop=True)  # 测试集
    data_df = pd.merge(train_df, test_df, how='outer').reset_index(drop=True)
    data_df['before30'] = data_df['amount'].shift(-30)
    data_df['before172'] = data_df['amount'].shift(-172)
    train_df = data_df.loc[data_df['date'] <= train_last_date, :].reset_index(drop=True)
    test_df = data_df.loc[data_df['date'] > train_last_date, :].reset_index(drop=True)

    train_x = train_df.loc[:, select_frts + ['before30', 'before172']].copy()  # (1035, 90)
    train_y = train_df['amount']
    test_x = test_df.loc[:, select_frts + ['before30', 'before172']].copy()  # (30, 90)
    print(train_x.shape, train_y.shape, test_x.shape)
    lgb_train, lgb_test, sb, cv_scores, importance_list = lgb_cv(train_x, train_y, test_x)  # lgb_test.shape = (30,)
    lgb_test_A = [item if item > 0 else 0 for item in lgb_test]  # 取大于零的值
    result.loc[(result['post_id'] == 'A') & (result['periods'] == i), 'amount'] = lgb_test_A

    for i in range(18, 38):
        train_df = train_hour_df_B.loc[train_hour_df_B['periods'] == i, :].reset_index(drop=True)  # 训练集
        train_df = train_df.loc[(train_df['date'] > '2019-09-19') | (train_df['date'] < '2019-09-11'), :].reset_index(
            drop=True)
        train_df = train_df.loc[(train_df['date'] > '2019-12-31') | (train_df['date'] < '2019-11-22'), :].reset_index(
            drop=True)
        train_df = train_df.loc[train_df['date'] != '2018-10-17', :].reset_index(
            drop=True)
        # train_df = train_df.loc[(train_df['date'] >= '2020-04-01') | (train_df['date'] <= '2020-01-28'), :].reset_index(
        #     drop=True)
        test_df = test_hour_df_B.loc[test_hour_df_B['periods'] == i, :].reset_index(drop=True)  # 测试集
        train_x = train_df[select_frts].copy()
        train_y = train_df['amount']
        test_x = test_df[select_frts].copy()
        print(train_x.shape, train_y.shape, test_x.shape)
        lgb_train, lgb_test, sb, cv_scores, importance_list = lgb_cv(train_x, train_y, test_x)
        lgb_test_B = [item if item > 0 else 0 for item in lgb_test] # 取大于零的值
        result.loc[(result['post_id'] == 'B') & (result['periods'] == i), 'amount'] = lgb_test_B

    result['amount'] = result['amount'].fillna(value=0)
    result['amount'] = result['amount'].round(0).astype('int')
    result.to_csv('./result_periods1.txt', index=False)