# ## Multivariate time series forecasting

# ## ADD time features to our model
# def create_time_features(df,target=None):
#     """
#     Creates time series features from datetime index
#     """
#     df['date'] = df.index
#     df['hour'] = df['date'].dt.hour
#     df['dayofweek'] = df['date'].dt.dayofweek
#     df['quarter'] = df['date'].dt.quarter
#     df['month'] = df['date'].dt.month
#     df['year'] = df['date'].dt.year
#     df['dayofyear'] = df['date'].dt.dayofyear
#     df['sin_day'] = np.sin(df['dayofyear'])
#     df['cos_day'] = np.cos(df['dayofyear'])
#     df['dayofmonth'] = df['date'].dt.day
#     df['weekofyear'] = df['date'].dt.weekofyear
#     X = df.drop(['date'],axis=1)
#     if target:
#         y = df[target]
#         X = X.drop([target],axis=1)
#         return X, y

#     return X


# X_train_df, y_train = create_time_features(df_training, target='pollution_today')
# X_test_df, y_test = create_time_features(df_test, target='pollution_today')
# scaler = StandardScaler()
# scaler.fit(X_train_df) #No cheating, never scale on the training+test!
# X_train = scaler.transform(X_train_df)
# X_test = scaler.transform(X_test_df)

# X_train_df = pd.DataFrame(X_train,columns=X_train_df.columns)
# X_test_df = pd.DataFrame(X_test,columns=X_test_df.columns)


# Required
#
# X = (Days, features)
# Y = (Days,)
#
# But we don't have time series features for each districts. Therefore number of features is 1 (covid cases).
#


# ### Linear models

import numpy as np
from datetime import datetime
# from fbprophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn import linear_model, svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

import lightgbm as lgb

def BaysianRegression(df, df_training, df_test):
    n_regions = df_training.shape[1]
    yhat = []
    for col in range(n_regions):
        reg = linear_model.BayesianRidge()

        today = df_training.iloc[:-1, col:col + 1]
        tomorrow = df_training.iloc[1:, col:col + 1]
        reg.fit(today, tomorrow)
        yhat.append(reg.predict(df_test.iloc[:, col:col + 1]))

    yhat = np.squeeze(np.array(yhat)).T
    return yhat

def Lasso(df, df_training, df_test):
    n_regions = df_training.shape[1]

    yhat = []
    for col in range(n_regions):
        reg = linear_model.Lasso(alpha=0.1)
        today = df_training.iloc[:-1, col:col + 1]
        tomorrow = df_training.iloc[1:, col:col + 1]
        reg.fit(today, tomorrow)
        yhat.append(reg.predict(df_test.iloc[:, col:col + 1]))

    yhat = np.squeeze(np.array(yhat)).T
    return yhat
# ### Tree models

def Randomforest(df, df_training, df_test):
    n_regions = df_training.shape[1]

    yhat = []
    for col in range(n_regions):
        reg = RandomForestRegressor(max_depth=2, random_state=0)
        reg.fit(df_training.iloc[:-1, col:col + 1], df_training.iloc[1:, col:col + 1])
        yhat.append(reg.predict(df_test.iloc[:, col:col + 1]))

    yhat = np.squeeze(np.array(yhat)).T
    return yhat

def XGBoost(df, df_training, df_test):
    n_regions = df_training.shape[1]
    yhat = []
    for col in range(n_regions):
        reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        reg.fit(df_training.iloc[:-1, col:col + 1], df_training.iloc[1:, col:col + 1], verbose=False)
        yhat.append(reg.predict(df_test.iloc[:, col:col + 1]))

    yhat = np.squeeze(np.array(yhat)).T
    return yhat


def Lightgbm(df, df_training, df_test):
    # A tree gradient boosting model by [microsoft](https://github.com/microsoft/LightGBM)
    n_regions = df_training.shape[1]

    yhat = []
    for col in range(n_regions):
        reg = lgb.LGBMRegressor()
        reg.fit(df_training.iloc[:-1, col:col + 1], df_training.iloc[1:, col:col + 1])
        yhat.append(reg.predict(df_test.iloc[:, col:col + 1]))

    yhat = np.squeeze(np.array(yhat)).T
    return yhat

def SVM_RBF(df, df_training, df_test):
    n_regions = df_training.shape[1]
    yhat = []
    for col in range(n_regions):
        reg = svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        reg.fit(df_training.iloc[:-1, col:col + 1], df_training.iloc[1:, col:col + 1])
        yhat.append(reg.predict(df_test.iloc[:, col:col + 1]))

    yhat = np.squeeze(np.array(yhat)).T
    return yhat

def Kneighbors(df, df_training, df_test):
    n_regions = df_training.shape[1]
    yhat = []
    for col in range(n_regions):
        reg = KNeighborsRegressor(n_neighbors=2)
        reg.fit(df_training.iloc[:-1, col:col + 1], df_training.iloc[1:, col:col + 1])
        yhat.append(reg.predict(df_test.iloc[:, col:col + 1]))

    yhat = np.squeeze(np.array(yhat)).T
    return yhat
