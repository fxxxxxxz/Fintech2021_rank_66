import pandas as pd
import numpy as np
import os
import lightgbm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.plot import plot_components_plotly

train_df=pd.read_csv('./train_v2.csv')
test_df=pd.read_csv('./test_v2_periods.csv')#按0.5h计算
test_day=pd.read_csv('./test_v2_day.csv')#按天计算
wkd_df=pd.read_csv('./wkd_v2.csv')
wkd_df=wkd_df.rename(columns={'ORIG_DT':'date'})
train_df=train_df.merge(wkd_df)


def compute_feature(df):
    df['WKD_TYP_CD']=df['WKD_TYP_CD'].map({'WN':0,'SN': 1, 'NH': 2, 'SS': 3, 'WS': 4})
    df['date']=pd.to_datetime(df['date'])
    df['dayofweek']=df['date'].dt.dayofweek+1
    df['day']=df['date'].dt.day
    df['month']=df['date'].dt.month
    df['year']=df['date'].dt.year
    # df.drop(['date','post_id'],axis=1,inplace=True)
    return df

#train
tmp=train_df[['date','post_id','amount']].groupby(['date','post_id'],sort=False).agg('sum')
train_day_df=pd.DataFrame(tmp).reset_index()
train_day_df_A=train_day_df[train_day_df['post_id']=='A'].reset_index(drop=True)
train_day_df_B=train_day_df[train_day_df['post_id']=='B'].reset_index(drop=True)
train_day_df_A=train_day_df_A.merge(wkd_df)
train_day_df_B=train_day_df_B.merge(wkd_df)
train_day_df_A=compute_feature(train_day_df_A)
train_day_df_B=compute_feature(train_day_df_B)

#test
tmp=test_df[['date','post_id']].groupby(['date','post_id'],sort=False).agg('sum')
test_day_df=pd.DataFrame(tmp).reset_index()
test_day_df_A=test_day_df[test_day_df['post_id']=='A'].reset_index(drop=True)
test_day_df_B=test_day_df[test_day_df['post_id']=='B'].reset_index(drop=True)
test_day_df_A=test_day_df_A.merge(wkd_df)
test_day_df_B=test_day_df_B.merge(wkd_df)
test_day_df_A=compute_feature(test_day_df_A)
test_day_df_B=compute_feature(test_day_df_B)

YDQ = pd.DataFrame({#元旦假期前的最后一个工作日及之前4天
  'holiday': 'YDQ',
  'ds': pd.to_datetime(['2018-12-29', '2020-12-31']),
  'lower_window': -15,
  'upper_window': 0,
})
YD = pd.DataFrame({ #从元旦假期的第一天开始
  'holiday': 'YD',
  'ds': pd.to_datetime(['2018-12-30', '2021-01-01']),
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
  'lower_window': -10,
  'upper_window': 0,
})
CJ = pd.DataFrame({
  'holiday': 'CJ',
  'ds': pd.to_datetime(['2018-02-16', '2019-02-05', '2020-01-25', '2021-02-12']),
  'lower_window': -10,
  'upper_window': 20,
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
YQ = pd.DataFrame({ #新冠疫情
  'holiday': 'YQ',
  'ds': pd.to_datetime(['2020-01-20']),
  'lower_window': 0,
  'upper_window': 70,
})


holidays = pd.concat((YQ, YDQ, YD, QR, FN, QM, LD, ET, GQ, PAY, SD, LB, CX, XN, CJ, CW, YX, DW, QX, ZY, ZQ, CY, DZ))


train = train_day_df_A.loc[:,['date', 'amount']]
train = train.rename(columns={'date':'ds', 'amount':'y'})
train['ds'] = pd.to_datetime(train['ds'])
train.loc[train['y'] == 0, 'y'] = None
train.loc[train['y'] > train['y'].quantile(q=0.998), "y"] = None
# train.loc[(train['ds'] > '2020-01-29') & (train['ds'] < '2020-04-01'), "y"] = None
train.loc[(train['ds'] >= '2019-12-25') & (train['ds'] <= '2020-01-05'), "y"] = None
train.loc[train['ds'] == '2019-11-25', "y"] = None
train.loc[train['ds'] == '2019-12-09', "y"] = None
m = Prophet(holidays=holidays)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(train)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
# fig1 = m.plot(forecast)
fig2 = plot_plotly(m, forecast)
fig2.show()
# m.plot_components(forecast)
forecast.to_csv('./features/feature_A.csv')

train = train_day_df_B.loc[:,['date', 'amount']]
train = train.rename(columns={'date':'ds', 'amount':'y'})
train['ds'] = pd.to_datetime(train['ds'])
train.loc[train['y'] == 0, 'y'] = None
train.loc[train['y'] < 130, 'y'] = None
train.loc[train['y'] > train['y'].quantile(q=0.998), "y"] = None
# train.loc[(train['ds'] > '2020-01-29') & (train['ds'] < '2020-04-01'), "y"] = None
train.loc[(train['ds'] >= '2019-09-11') & (train['ds'] <= '2019-09-19'), "y"] = None
train.loc[(train['ds'] >= '2019-11-22') & (train['ds'] <= '2019-12-31'), "y"] = None
train.loc[train['ds'] == '2018-10-17', "y"] = None
# train.loc[(train['ds'] >= '2020-01-06') & (train['ds'] <= '2020-01-23'), "y"] = None
m = Prophet(holidays=holidays)
m.add_seasonality(name='monthly', period=30.5, fourier_order=3)
m.fit(train)
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig3 = plot_plotly(m, forecast)
fig3.show()
# fig4 = plot_components_plotly(m, forecast)
# fig4.show()
# fig5 = m.plot(forecast)
# m.plot_components(forecast)
forecast.to_csv('./features/feature_B.csv')