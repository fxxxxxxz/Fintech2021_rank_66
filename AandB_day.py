import pandas as pd
import numpy as np
import os
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import graphviz

train_df = pd.read_csv('./train_v2.csv')
test_df = pd.read_csv('./test_v2_periods.csv')
test_day = pd.read_csv('./test_v2_day.csv')
wkd_df = pd.read_csv('./wkd_v2.csv')
wkd_df = wkd_df.rename(columns={'ORIG_DT': 'date'})
train_df = train_df.merge(wkd_df)


# 处理特征
def compute_feature(df):
    df['WKD_TYP_CD'] = df['WKD_TYP_CD'].map({'WN': 0, 'SN': 1, 'NH': 1, 'SS': 1, 'WS': 0})
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.dayofweek + 1
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    season_dict = {  # 冬夏长，春秋短
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


# train
tmp = train_df[['date', 'post_id', 'amount']].groupby(['date', 'post_id'], sort=False).agg('sum')
train_day_df = pd.DataFrame(tmp).reset_index()
train_day_df_A = train_day_df[train_day_df['post_id'] == 'A'].reset_index(drop=True)
train_day_df_B = train_day_df[train_day_df['post_id'] == 'B'].reset_index(drop=True)
train_day_df_A = train_day_df_A.merge(wkd_df)
train_day_df_B = train_day_df_B.merge(wkd_df)
train_day_df_A = compute_feature(train_day_df_A)
train_day_df_B = compute_feature(train_day_df_B)

# test
tmp = test_df[['date', 'post_id']].groupby(['date', 'post_id'], sort=False).agg('sum')
test_day_df = pd.DataFrame(tmp).reset_index()
test_day_df_A = test_day_df[test_day_df['post_id'] == 'A'].reset_index(drop=True)
test_day_df_B = test_day_df[test_day_df['post_id'] == 'B'].reset_index(drop=True)
test_day_df_A = test_day_df_A.merge(wkd_df)
test_day_df_B = test_day_df_B.merge(wkd_df)
test_day_df_A = compute_feature(test_day_df_A)
test_day_df_B = compute_feature(test_day_df_B)

feature_A = pd.read_csv('./features/feature_A.csv')
feature_B = pd.read_csv('./features/feature_B.csv')
feature_A = feature_A.rename(columns={'ds': 'date'})
feature_A['date'] = pd.to_datetime(feature_A['date'])
feature_A.drop(['Unnamed: 0'], axis=1, inplace=True)
feature_B = feature_B.rename(columns={'ds': 'date'})
feature_B['date'] = pd.to_datetime(feature_B['date'])
feature_B.drop(['Unnamed: 0'], axis=1, inplace=True)

train_day_df_A = train_day_df_A.merge(feature_A, how='left', on='date')
test_day_df_A = test_day_df_A.merge(feature_A, how='left', on='date')
train_day_df_B = train_day_df_B.merge(feature_B, how='left', on='date')
test_day_df_B = test_day_df_B.merge(feature_B, how='left', on='date')

# 训练集和测试集
print(train_day_df_A.shape, test_day_df_A.shape)  # (1035, 6) (30, 5)


def lgb_cv_A(train_x, train_y, test_x):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    folds = 10
    seed = 2021
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
            'metrics': 'l1',
            # 'num_leaves': 2 ** 6-1,
            # 'feature_fraction': 0.8,
            # 'bagging_fraction': 0.8,
            'min_data_in_leaf': 5,
            'learning_rate': 0.01,
            'seed': 2021,
            'nthread': 2,
            'verbose': -1,
        }
        num_round = 4000
        early_stopping_rounds = 100
        if test_matrix:
            model = lightgbm.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=200,
                                   # feval=tpr_eval_score,
                                   early_stopping_rounds=early_stopping_rounds
                                   )
            print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                                                                  key=lambda x: x[1], reverse=True))[:10]))
            importance_list = [x[0] for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                                                         key=lambda x: x[1], reverse=True))]
            # print(importance_list)
            pre = model.predict(te_x, num_iteration=model.best_iteration)  #
            pred = model.predict(test_x, num_iteration=model.best_iteration)  #
            # ax = lightgbm.plot_tree(model, tree_index=0, figsize=(255, 255), show_info=['split_gain'])
            # plt.show()
            train[test_index] = pre
            test_pre[i, :] = pred
            cv_scores.append(mean_squared_error(te_y, pre))
            cv_rounds.append(model.best_iteration)
            test_pre_all[i, :] = pred
        #
        print("cv_score is:", cv_scores)
    use_mean = True
    if use_mean:
        test[:] = test_pre.mean(axis=0)
    else:
        pass
    #
    print("val_mean:", np.mean(cv_scores))
    print("val_std:", np.std(cv_scores))
    return train, test, test_pre_all, np.mean(cv_scores), importance_list


def lgb_cv_B(train_x, train_y, test_x):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    folds = 10
    seed = 2021
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
            'metrics': 'l1',
            # 'num_leaves': 2 ** 6-1,
            # 'feature_fraction': 0.8,
            # 'bagging_fraction': 0.8,
            'min_data_in_leaf': 5,
            'learning_rate': 0.01,
            'seed': 2021,
            'nthread': 2,
            'verbose': -1,
        }
        num_round = 4000
        early_stopping_rounds = 100
        if test_matrix:
            model = lightgbm.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=200,
                                   # feval=tpr_eval_score,
                                   early_stopping_rounds=early_stopping_rounds
                                   )
            print("\n".join(("%s: %.2f" % x) for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                                                                  key=lambda x: x[1], reverse=True))[:10]))
            importance_list = [x[0] for x in list(sorted(zip(predictors, model.feature_importance("gain")),
                                                         key=lambda x: x[1], reverse=True))]
            # print(importance_list)
            pre = model.predict(te_x, num_iteration=model.best_iteration)  #
            pred = model.predict(test_x, num_iteration=model.best_iteration)  #

            train[test_index] = pre
            test_pre[i, :] = pred
            cv_scores.append(mean_squared_error(te_y, pre))
            cv_rounds.append(model.best_iteration)
            test_pre_all[i, :] = pred
        #
        print("cv_score is:", cv_scores)
    use_mean = True
    if use_mean:
        test[:] = test_pre.mean(axis=0)
    else:
        pass
    #
    print("val_mean:", np.mean(cv_scores))
    print("val_std:", np.std(cv_scores))
    return train, test, test_pre_all, np.mean(cv_scores), importance_list


if __name__ == "__main__":
    select_frts = ['WKD_TYP_CD', 'year', 'month', 'day', 'dayofweek', 'season', 'tenDay',
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
    cat_features = ['WKD_TYP_CD', 'year', 'month', 'day', 'dayofweek', 'season', 'tenDay']

    train_day_df_A[cat_features] = train_day_df_A[cat_features].astype('category')
    test_day_df_A[cat_features] = test_day_df_A[cat_features].astype('category')
    train_day_df_B[cat_features] = train_day_df_B[cat_features].astype('category')
    test_day_df_B[cat_features] = test_day_df_B[cat_features].astype('category')

    train_df = train_day_df_A  # 训练集A
    test_df = test_day_df_A  # 测试集A
    train_df = train_df.loc[(train_df['date'] >= '2020-04-01') | (train_df['date'] <= '2020-01-28'), :].reset_index(
        drop=True)
    train_df = train_df.loc[(train_df['date'] != '2019-12-09') & (train_df['date'] != '2019-11-25'), :].reset_index(drop=True)
    train_df = train_df.loc[(train_df['date'] > '2020-01-05') | (train_df['date'] < '2019-12-25'), :].reset_index(drop=True)
    train_df = train_df.loc[train_df['amount'] != 0, :].reset_index(drop=True)
    train_x = train_df[select_frts].copy()
    train_y = train_df['amount']
    test_x = test_df[select_frts].copy()
    print(train_x.shape, train_y.shape, test_x.shape)
    lgb_train, lgb_test, sb, cv_scores, importance_list = lgb_cv_A(train_x, train_y, test_x)
    lgb_test_A = [item if item > 0 else 0 for item in lgb_test]

    #
    train_df = train_day_df_B  # 训练集B
    test_df = test_day_df_B  # 测试集B
    train_df = train_df.loc[(train_df['date'] >= '2020-04-01') | (train_df['date'] <= '2020-01-28'), :].reset_index(
        drop=True)
    train_df = train_df.loc[(train_df['date'] > '2019-09-19') | (train_df['date'] < '2019-09-11'), :].reset_index(
        drop=True)
    train_df = train_df.loc[(train_df['date'] > '2019-12-31') | (train_df['date'] < '2019-11-22'), :].reset_index(
        drop=True)
    train_df = train_df.loc[train_df['date'] != '2018-10-17', :].reset_index(
        drop=True)
    train_x = train_df[select_frts].copy()
    train_y = train_df['amount']
    test_x = test_df[select_frts].copy()
    print(train_x.shape, train_y.shape, test_x.shape)
    lgb_train, lgb_test, sb, cv_scores, importance_list = lgb_cv_B(train_x, train_y, test_x)
    lgb_test_B = [item if item > 0 else 0 for item in lgb_test]
    print(np.mean(lgb_test_A), np.sum(lgb_test_A), np.mean(lgb_test_B), np.sum(lgb_test_B))
    # # 使用LGB的预测值
    pre_A = np.array(lgb_test_A)
    pre_B = np.array(lgb_test_B)
    # 使用prophet的回归值
    # pre_A = test_day_df_A['yhat'].values
    # pre_B = test_day_df_B['yhat'].values
    test_day = pd.read_csv('./test_v2_day.csv')  # 按天计算
    pre_day = []
    for i in range(31):
        pre_day.append(pre_A[i])
        pre_day.append(pre_B[i])
    test_day['amount'] = pre_day

    f = open('./LGB/day/result_day.txt', 'w')
    f.write('Date' + ',' + 'Post_id' + ',' + 'Predict_amount' + '\n')
    for _, date, post_id, amount in test_day.itertuples():
        f.write(date + ',' + post_id + ',' + str(int(amount)) + '\n')
    f.close()
