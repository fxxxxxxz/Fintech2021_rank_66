import pandas as pd
import numpy as np
import os
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import itertools
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation

YDQ = pd.DataFrame({#元旦假期前的最后一个工作日及之前4天
  'holiday': 'YDQ',
  'ds': pd.to_datetime(['2018-12-29', '2020-12-31']),
  'lower_window': -5,
  'upper_window': 0,
})
YD = pd.DataFrame({ #从元旦假期的第一天开始
  'holiday': 'YD',
  'ds': pd.to_datetime(['2017-12-30', '2018-12-30', '2021-01-01']),
  'lower_window': 0,
  'upper_window': 2,
})
QR = pd.DataFrame({
  'holiday': 'QR',
  'ds': pd.to_datetime(['2018-02-14', '2019-02-14', '2020-02-14', '2021-02-14']),
  'lower_window': -1,
  'upper_window': 0,
})
FN = pd.DataFrame({
  'holiday': 'FN',
  'ds': pd.to_datetime(['2018-03-08', '2019-03-08', '2020-03-08', '2021-03-08']),
  'lower_window': 0,
  'upper_window': 0,
})
QM = pd.DataFrame({
  'holiday': 'QM',
  'ds': pd.to_datetime(['2018-04-05', '2019-04-05', '2020-04-04', '2021-04-04']),
  'lower_window': -1,
  'upper_window': 3,
})
LD = pd.DataFrame({
  'holiday': 'LD',
  'ds': pd.to_datetime(['2018-05-01', '2019-05-01', '2020-05-01', '2021-05-01']),
  'lower_window': -1,
  'upper_window': 3,
})
ET = pd.DataFrame({
  'holiday': 'ET',
  'ds': pd.to_datetime(['2018-06-01', '2019-06-01', '2020-06-01', '2021-06-01']),
  'lower_window': 0,
  'upper_window': 0,
})
GQ = pd.DataFrame({
  'holiday': 'GQ',
  'ds': pd.to_datetime(['2018-10-01', '2019-10-01', '2020-10-01', '2021-10-01']),
  'lower_window': 0,
  'upper_window': 7,
})
PAY = pd.DataFrame({
  'holiday': 'PAY',
  'ds': pd.to_datetime(['2018-12-24', '2019-12-24', '2020-12-24', '2021-12-24']),
  'lower_window': 0,
  'upper_window': 0,
})
SD = pd.DataFrame({
  'holiday': 'SD',
  'ds': pd.to_datetime(['2018-12-25', '2019-12-25', '2020-12-25', '2021-12-25']),
  'lower_window': -1,
  'upper_window': 0,
})
LB = pd.DataFrame({
  'holiday': 'LB',
  'ds': pd.to_datetime(['2018-01-24', '2019-01-13', '2020-01-02', '2021-01-20']),
  'lower_window': 0,
  'upper_window': 0,
})
CX = pd.DataFrame({
  'holiday': 'CX',
  'ds': pd.to_datetime(['2018-02-15', '2019-02-04', '2020-01-24', '2021-02-11']),
  'lower_window': -2,
  'upper_window': 0,
})
XN = pd.DataFrame({
  'holiday': 'XN',
  'ds': pd.to_datetime(['2018-02-08', '2019-01-28', '2020-01-17', '2021-02-04']),
  'lower_window': -5,
  'upper_window': 0,
})
CJ = pd.DataFrame({
  'holiday': 'CJ',
  'ds': pd.to_datetime(['2018-02-16', '2019-02-05', '2020-01-25', '2021-02-12']),
  'lower_window': -5,
  'upper_window': 7,
})
CW = pd.DataFrame({
  'holiday': 'CW',
  'ds': pd.to_datetime(['2018-02-20', '2019-02-09', '2020-01-29', '2021-02-16']),
  'lower_window': 0,
  'upper_window': 0,
})
YX = pd.DataFrame({
  'holiday': 'YX',
  'ds': pd.to_datetime(['2018-03-02', '2019-02-19', '2020-02-08', '2021-02-26']),
  'lower_window': -1,
  'upper_window': 1,
})
DW = pd.DataFrame({
  'holiday': 'DW',
  'ds': pd.to_datetime(['2018-06-18', '2019-06-07', '2020-06-25', '2021-06-14']),
  'lower_window': -1,
  'upper_window': 3,
})
QX = pd.DataFrame({
  'holiday': 'QX',
  'ds': pd.to_datetime(['2018-08-17', '2019-08-07', '2020-08-25', '2021-08-14']),
  'lower_window': 0,
  'upper_window': 0,
})
ZY = pd.DataFrame({
  'holiday': 'ZY',
  'ds': pd.to_datetime(['2018-08-25', '2019-08-15', '2020-09-02', '2021-08-22']),
  'lower_window': -1,
  'upper_window': 0,
})
ZQ = pd.DataFrame({
  'holiday': 'ZQ',
  'ds': pd.to_datetime(['2018-09-24', '2019-09-13', '2020-10-01', '2021-09-21']),
  'lower_window': -1,
  'upper_window': 3,
})
CY = pd.DataFrame({
  'holiday': 'CY',
  'ds': pd.to_datetime(['2018-10-17', '2019-10-07', '2020-10-25', '2021-10-14']),
  'lower_window': 0,
  'upper_window': 0,
})
DZ = pd.DataFrame({
  'holiday': 'DZ',
  'ds': pd.to_datetime(['2018-12-22', '2019-12-22', '2020-12-21', '2021-12-21']),
  'lower_window': 0,
  'upper_window': 0,
})


holidays = pd.concat((YDQ, YD, QR, FN, QM, LD, ET, GQ, PAY, SD, LB, CX, XN, CJ, CW, YX, DW, QX, ZY, ZQ, CY, DZ))

#处理特征
def compute_feature(df):
    df['WKD_TYP_CD']=df['WKD_TYP_CD'].map({'WN':0,'SN': 1, 'NH': 2, 'SS': 3, 'WS': 4})
    df['date']=pd.to_datetime(df['date'])
    df['dayofweek']=df['date'].dt.dayofweek+1
    df['day']=df['date'].dt.day
    df['month']=df['date'].dt.month
    df['year']=df['date'].dt.year
    # df.drop(['date','post_id'],axis=1,inplace=True)
    return df


train_df = pd.read_csv('D:/自主学习/招行/train_v2.csv')
wkd_df = pd.read_csv('D:/自主学习/招行/wkd_v2.csv')
wkd_df = wkd_df.rename(columns={'ORIG_DT': 'date'})
train_df = train_df.merge(wkd_df)
#
train_df['amount'] = train_df['amount']
train_hour_df_A_cls = train_df[train_df['post_id'] == 'A'].reset_index(drop=True)
train_hour_df_B = train_df[train_df['post_id'] == 'B'].reset_index(drop=True)
#
name_A = ['A' + str(i) for i in range(2, 14)]
train_hour_df_A = train_hour_df_A_cls[train_hour_df_A_cls['biz_type'] == 'A1'].reset_index(drop=True)
new_train_amount = train_hour_df_A['amount'].values
for Ai in name_A:
  tmp = train_hour_df_A_cls[train_hour_df_A_cls['biz_type'] == Ai].reset_index(drop=True)
  new_train_amount += tmp['amount'].values
#
train_hour_df_A['amount'] = new_train_amount
train_hour_df_A.drop(['biz_type'], axis=1, inplace=True)
train_hour_df_B.drop(['biz_type'], axis=1, inplace=True)
train_hour_df_A = compute_feature(train_hour_df_A)
train_hour_df_B = compute_feature(train_hour_df_B)

# A
for i in range(1, 49):
    temp = train_hour_df_A.loc[train_hour_df_A.loc[:,'periods'] == i, :]
    train = pd.DataFrame({
      "ds": pd.to_datetime(temp.loc[:, 'date']),
      "y": temp['amount']
    })
    if train['y'].sum() > 1000:
        train.loc[train['y'] == 0, 'y'] = None
        train.loc[(train['ds'] > '2020-02-01') & (train['ds'] < '2020-04-01'), "y"] = None
        train.loc[train['y'] > train['y'].quantile(q=0.995), "y"] = None
        train.loc[train['ds'] == '2019-12-09', "y"] = None

    model = Prophet(growth='linear', holidays=holidays)
    model.fit(train)
    future = model.make_future_dataframe(periods=70)
    forecast = model.predict(future)
    # fig1 = model.plot(forecast)
    # fig1 = plot_plotly(model, forecast)
    # fig1.show()
    # fig2 = model.plot_components(forecast)
    # del model
    forecast.to_csv('D:/自主学习/招行/features/feature_' + 'A' + '_' + str(i) + '.csv')

# B
for i in range(1, 49):
    temp = train_hour_df_B.loc[train_hour_df_B.loc[:,'periods'] == i, :]
    train = pd.DataFrame({
      "ds": pd.to_datetime(temp.loc[:, 'date']),
      "y": temp['amount']
    })
    if train['y'].sum() > 1000:
        train.loc[train['y'] == 0, 'y'] = None
        train.loc[train['y'] < train['y'].quantile(q=0.05), "y"] = None
    #     # train.loc[(train['ds'] > '2020-01-01') & (train['ds'] < '2020-04-01'), "y"] = None
    #     train.loc[train['y'] > train['y'].quantile(q=0.99), "y"] = None
    #     # train.loc[(train['ds'] >= '2020-04-01') & (train['ds'] <= '2020-04-15'), "y"] *= 1.25
    # # train.loc[(train['y'] < 500) | (train['y'] > 7500), 'y'] = None

    model = Prophet(growth='linear', holidays=holidays)
    model.fit(train)
    future = model.make_future_dataframe(periods=70)
    forecast = model.predict(future)
    # fig1 = model.plot(forecast)
    # fig1 = plot_plotly(model, forecast)
    # fig1.show()
    # fig2 = model.plot_components(forecast)
    # del model
    forecast.to_csv('D:/自主学习/招行/features/feature_' + 'B' + '_' + str(i) + '.csv')







