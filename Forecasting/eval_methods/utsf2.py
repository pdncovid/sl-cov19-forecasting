# ## Univariate-time-series-forecasting

# In this section we will focus on time series forecasting methods capable of only looking at the target variable.
# This means no other regressors (more variables) can be added into the model.


# ================================================================================### Simple Exponential Smoothing (SES)

# The Simple Exponential Smoothing (SES) method models the next time step as an exponentially weighted linear function
# of observations at prior time steps. This method expects our time series to be non stationary in order to perform
# adecuately (no trend or seasonality)
import warnings

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import numpy as np
import pmdarima as pm

warnings.filterwarnings(
    "ignore")  # We will use deprecated models of statmodels which throw a lot of warnings to use more modern ones

from tqdm import tqdm, tqdm_notebook

# NOPE
def SES(df, df_training, df_test):
    # Walk throught the test data, training and predicting 1 day ahead for all the test data
    index = len(df_training)
    n_regions = df_training.shape[1]
    yhat = [[]] * n_regions

    temp_train = df.iloc[:len(df_training), :]

    for col in range(n_regions):
        inp = temp_train.iloc[:, col].values
        for t in tqdm(range(len(df_test))):
            model = SimpleExpSmoothing(inp)
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(inp), end=len(inp))
            inp = np.concatenate([inp, predictions])
            yhat[col] = yhat[col] + [predictions]

    yhat = np.squeeze(np.array(yhat)).T
    return yhat

# NOPE
def HWES(df, df_training, df_test):
    # ### Holt Winter’s Exponential Smoothing (HWES)

    # (https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/) or also known as triple exponential smoothing

    index = len(df_training)
    n_regions = df_training.shape[1]
    yhat = [[]] * df_training.shape[1]

    temp_train = df.iloc[:len(df_training), :]
    for col in range(n_regions):
        inp = temp_train.iloc[:, col].values
        for t in tqdm(range(len(df_test))):
            model = ExponentialSmoothing(inp)
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(inp), end=len(inp))
            inp = np.concatenate([inp, predictions])
            yhat[col] = yhat[col] + [predictions]

    yhat = np.squeeze(np.array(yhat)).T
    return yhat


def mAR(df, df_training, df_test):
    # ### Autoregression (AR)
    # The autoregression (AR) method models the next step in the sequence as a linear function of the observations at
    # prior time steps. Parameters of the model:
    # Number of AR (Auto-Regressive) terms (p):__ p is the parameter associated with the auto-regressive aspect of the
    # model, which incorporates past values i.e lags of dependent variable. For instance if p is 5, the predictors for
    # x(t) will be x(t-1)….x(t-5).
    from statsmodels.tsa.ar_model import AR

    index = len(df_training)
    n_regions = df_training.shape[1]

    yhat = [[]] * df_training.shape[1]
    temp_train = df.iloc[:len(df_training), :]
    for col in range(n_regions):
        inp = temp_train.iloc[:, col].values
        for t in tqdm(range(len(df_test))):
            model = AR(inp)
            model_fit = model.fit()
            predictions = model_fit.predict(start=len(inp), end=len(inp))
            inp = np.concatenate([inp, predictions])
            yhat[col] = yhat[col] + [predictions]

    yhat = np.squeeze(np.array(yhat)).T
    return yhat


def MA(df, df_training, df_test):
    # ### Moving Average (MA)

    # The Moving Average (MA) method models the next step in the sequence as the average of a window of observations at
    # prior time steps. Parameters of the model:
    # - __Number of MA (Moving Average) terms (q):__ q is size of the moving average part window of the model
    # i.e. lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will
    # be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.

    from statsmodels.tsa.arima_model import ARMA

    index = len(df_training)
    n_regions = df_training.shape[1]
    yhat = [[]] * df_training.shape[1]
    temp_train = df.iloc[:len(df_training), :]
    for col in range(n_regions):
        inp = temp_train.iloc[:, col].values
        for t in tqdm(range(len(df_test))):
            model = ARMA(inp, order=(0, 1))
            model_fit = model.fit(disp=False)
            predictions = model_fit.predict(start=len(inp), end=len(inp))
            inp = np.concatenate([inp, predictions])
            yhat[col] = yhat[col] + [predictions]

    yhat = np.squeeze(np.array(yhat)).T
    return yhat


def ARIMA(df, df_training, df_test):
    # ### Autoregressive integrated moving average (ARIMA)

    # In an ARIMA model there are 3 parameters that are used to help model the major aspects of a times series:
    # seasonality, trend, and noise. These parameters are labeled p,d,and q.
    #
    # * Number of AR (Auto-Regressive) terms (p): p is the parameter associated with the auto-regressive aspect of the
    # model, which incorporates past values i.e lags of dependent variable. For instance if p is 5, the predictors for
    # x(t) will be x(t-1)….x(t-5).
    # * Number of Differences (d): d is the parameter associated with the integrated part of the model, which effects
    # the amount of differencing to apply to a time series.
    # * Number of MA (Moving Average) terms (q): q is size of the moving average part window of the model i.e. lagged
    # forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5)
    # where e(i) is the difference between the moving average at ith instant and actual value.
    #
    # **Tuning ARIMA parameters**
    #
    # Non stationarity series will require level of differencing (d) >0 in ARIMA
    # Select the lag values for the Autoregression (AR) and Moving Average (MA) parameters, p and q respectively, using
    # PACF, ACF plots
    # AUTOARIMA
    #
    # Note: A problem with ARIMA is that it does not support seasonal data. That is a time series with a repeating
    # cycle. ARIMA expects data that is either not seasonal or has the seasonal component removed, e.g. seasonally
    # adjusted via methods such as seasonal differencing.

    # In[ ]:

    # ARIMA example
    from statsmodels.tsa.arima_model import ARIMA
    from sklearn.metrics import mean_squared_error
    from math import sqrt

    # Walk throught the test data, training and predicting 1 day ahead for all the test data
    index = len(df_training)
    yhat = [[]] * df_training.shape[1]
    n_regions = df_training.shape[1]
    temp_train = df.iloc[:len(df_training), :]
    for col in range(n_regions):
        inp = temp_train.iloc[:, col].values
        for t in tqdm(range(len(df_test))):
            model = ARIMA(inp, order=(1, 0, 0))
            model_fit = model.fit(disp=False)
            predictions = model_fit.predict(start=len(inp), end=len(inp), dynamic=False)
            inp = np.concatenate([inp, predictions])
            yhat[col] = yhat[col] + [predictions]

    yhat = np.squeeze(np.array(yhat)).T
    return yhat


def SARIMA(df, df_training, df_test):
    # ### Seasonal Autoregressive Integrated Moving-Average (SARIMA)
    # Seasonal Autoregressive Integrated Moving Average, SARIMA or Seasonal ARIMA, is an extension of ARIMA that
    # explicitly supports univariate time series data with a seasonal component.
    #
    # It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA)
    # for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.
    #
    # __Trend Elements:__
    #
    # There are three trend elements that require configuration. They are the same as the ARIMA model, specifically:
    #
    # - p: Trend autoregression order.
    # - d: Trend difference order.
    # - q: Trend moving average order.
    #
    # __Seasonal Elements:__
    #
    # There are four seasonal elements that are not part of ARIMA that must be configured; they are:
    #
    # - P: Seasonal autoregressive order.
    # - D: Seasonal difference order.
    # - Q: Seasonal moving average order.
    # - m: The number of time steps for a single seasonal period. For example, an S of 12 for monthly data suggests a
    # yearly seasonal cycle.
    #
    # __SARIMA notation:__
    # SARIMA(p,d,q)(P,D,Q,m)


    yhat = [[]] * df_training.shape[1]
    n_regions = df_training.shape[1]

    temp_train = df.iloc[:len(df_training), :]
    for col in range(n_regions):
        inp = temp_train.iloc[:, col].values
        for t in tqdm(range(len(df_test))):
            model = SARIMAX(inp, order=(1, 0, 0), seasonal_order=(0, 0, 0, 3))
            model_fit = model.fit(disp=False)
            predictions = model_fit.predict(start=len(inp), end=len(inp), dynamic=False)
            inp = np.concatenate([inp, predictions])
            yhat[col] = yhat[col] + [predictions]


    yhat = np.squeeze(np.array(yhat)).T
    return yhat

def AutoSARIMA(df, df_training, df_test):
    # #### Auto - SARIMA
    #
    # [auto_arima documentation for selecting best model](https://www.alkaline-ml.com/pmdarima/tips_and_tricks.html)

    # In[ ]:

    # building the model
    autoModels = []
    for col in range(df_training.shape[1]):
        print(col)
        autoModel = pm.auto_arima(df_training.iloc[:, col], trace=True, error_action='ignore', suppress_warnings=True,
                                  seasonal=True, m=6, stepwise=True)
        autoModel.fit(df_training.iloc[:, col])
        autoModels.append(autoModel)

    # In[ ]:

    yhat = [[]] * df_training.shape[1]
    n_regions = df_training.shape[1]

    temp_train = df.iloc[:len(df_training), :]
    for col in range(n_regions):
        order = autoModels[col].order
        seasonalOrder = autoModels[col].seasonal_order
        inp = temp_train.iloc[:, col].values
        for t in tqdm(range(len(df_test))):
            model = SARIMAX(inp, order=order, seasonal_order=seasonalOrder)
            model_fit = model.fit(disp=False)
            predictions = model_fit.predict(start=len(inp), end=len(inp), dynamic=False)
            inp = np.concatenate([inp, predictions])
            yhat[col] = yhat[col] + [predictions]

    yhat = np.squeeze(np.array(yhat)).T
    return yhat

# plt.figure(figsize=(15, len(to_plot)))
# for i, tp in enumerate(to_plot):
#     plt.subplot(1 + len(to_plot) // 3, 3, i + 1)
#     plt.plot(df_test[tp].values, label='Original ' + str(tp))
#     plt.plot(yhat[:, list(df_test.columns).index(tp)], color='red', label='AR predicted ' + str(tp))
#     plt.legend()
