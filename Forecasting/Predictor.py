#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], '..'))

import pandas as pd  # Basic library for all of our dataset operations
import numpy as np
import tensorflow as tf
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
print(tf.__version__)


def undo_normalization(normalized_data, scalers):
    normalized_data = np.copy(normalized_data)
    if len(normalized_data.shape) == 2:
        normalized_data = np.expand_dims(normalized_data, 0)

    # print(f"DENORMALIZING; Norm Data: {normalized_data.shape} expected (samples, windowsize, region)")
    samples, windowsize, regions = normalized_data.shape

    normalized_data = scalers.inverse_transform(normalized_data.reshape((-1, regions)))
    normalized_data = normalized_data.reshape((samples, windowsize, regions))
    #     for i in range(len(scalers)):
    #         normalized_data[:,:,i] *= scalers[i]
    # #     normalized_data[normalized_data>10] = np.nan
    # #     normalized_data = np.exp(normalized_data)-1
    # #     print("NAN",np.isnan(normalized_data).sum())
    return normalized_data


def normalize_for_nn(data, given_scalers=None):
    data = np.copy(data)
    # print(f"NORMALIZING; Data: {data.shape} expected (regions, days)")
    #     data = np.log(data.astype('float32')+1)
    if given_scalers is None:
        scalers = MinMaxScaler()
        scalers.fit(data.T)
    else:
        scalers = given_scalers
    data = scalers.transform(data.T).T

    # scalers = []
    # scale = float(np.max(data[:,:]))
    # for i in range(data.shape[0]):
    #     if given_scalers is not None:
    #         scale = given_scalers[i]
    #     else:
    #         scale = float(np.max(data[i,:]))
    #     scalers.append(scale)
    #     data[i,:] /= scale

    return data, scalers


def per_million(data, population):
    # divide by population
    data_per_mio_capita = np.zeros_like(data)
    if population is None:
        population = np.ones((data.shape[0],)) * 1e6
    for i in range(len(population)):
        data_per_mio_capita[i, :] = data[i, :] / population[i] * 1e6
    return data_per_mio_capita


def get_data(filtered, normalize, data, dataf, population=None, lastndays=None):
    if not filtered:
        x, y = np.copy(data), np.copy(data)
    else:
        x, y = np.copy(dataf), np.copy(dataf)

    x = per_million(x, population)
    y = per_million(y, population)
    if lastndays is not None:
        x = x[-lastndays:]
        y = y[-lastndays:]
    if normalize:
        x, xs = normalize_for_nn(x, None if type(normalize) == bool else normalize)
        y, xs = normalize_for_nn(y, xs)
        return x.T, y.T, xs
    else:
        return x.T, y.T, None


def O_LPF(data, datatype, order, R_EIG_ratio, R_power, midpoint, corr, region_names, plot_freq, view, savepath=None):
    # print(f"Smoothing {data.shape}")
    midpoint = True

    if midpoint:
        R_EIG_ratio = 1.02
        R_power = 1
    else:
        R_EIG_ratio = 3
        R_power = 1

    if datatype == 'daily':
        data_sums = np.zeros(data.shape[0], )
        for i in range(data.shape[0]):
            data_sums[i] = np.sum(data[i, :])

    if midpoint:
        EIG_cons = R_EIG_ratio
        R_cons = 1
    else:
        R_cons = R_EIG_ratio
        EIG_cons = 1

    data = np.copy(data.T)
    n_regions = data.shape[1]
    cutoff_freqs = []  # to return the optimal cutoff frequencies
    # FILTERING:
    # Filter requirements.
    fs = 1
    nyq = 0.5 * fs

    def lowpass_filter(data, cutoff, fs, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data)
        return y.astype(np.float32)

    # DETERMINE THE RIGHT CUTOFF FREQUENCY
    # step = 0.01
    # cutoff_list = range(int(round(1 / step)))
    # cutoff_list = 0.1 * (np.array(list(cutoff_list)) + 5) / 100
    # print('cutoff_list=',cutoff_list)
    cutoff_list = np.linspace(0.011, 0.06, 50)

    sections = 10

    data_filtered = np.zeros_like(data)
    for i in range(n_regions):
        J_R = []
        J_eig = []
        J_tot = []
        for n in range(len(cutoff_list)):
            cutoff = cutoff_list[n]
            X = data[:, i]
            Y = lowpass_filter(X, cutoff, fs, order)

            # rescale filtered signal
            if datatype == 'daily':
                Y = data_sums[i] * Y / (np.sum(Y) if np.sum(Y) != 0 else 1)
                # Y[Y<0]=0
                # else:
                #   for n in range(len(Y)-1):
                #     if Y[n+1]-Y[n]<0:
                #       Y[n+1]=Y[n]
                Y = np.amax(X) * Y / (np.sum(Y) if np.sum(Y) != 0 else 1)

            if corr:
                J_R.append(np.mean(np.corrcoef(X, Y)))  # obtaining correlations
            else:
                J_R.append(1 / np.mean(np.square(X - Y)))  # obtaining error

            # obtaining power spectral densities
            X_freqs, X_psd = signal.welch(X)
            Y_freqs, Y_psd = signal.welch(Y)

            X_psd, Y_psd = np.log10(np.abs(X_psd) + 1e-10), np.log10(np.abs(Y_psd) + 1e-10)

            J0 = []

            sec_len = int(X_psd.shape[0] / sections)
            for k in range(sections):
                X_avg = np.mean(X_psd[k * sec_len:(k + 1) * sec_len])
                Y_avg = np.mean(Y_psd[k * sec_len:(k + 1) * sec_len])
                J0.append((k + 1) * np.abs(
                    X_avg - Y_avg))  # eigenvalue spread should increase as k increases for an ideal solution
            J_eig.append(np.sum(J0))
        # few assignments to get rid of errors
        J_eig = np.around(J_eig).astype(int) ** 2
        J_R = np.array(J_R) ** R_power
        J_eig[J_eig < 0] = 0
        J_EIG = J_eig / np.amax(J_eig) if np.amax(J_eig) != 0 else np.zeros_like(J_eig)
        J_Err = J_R / np.amax(J_R) if np.amax(J_R) != 0 else np.zeros_like(J_R)

        if midpoint:
            J_tot = 1 - np.abs(R_cons * (J_Err) - EIG_cons * (J_EIG))
        else:
            J_tot = R_cons * (J_Err) + EIG_cons * (J_EIG)

        J_tot = J_tot / np.amax(J_tot)
        idx = np.argmax(J_tot)
        Y = lowpass_filter(X, cutoff_list[idx], fs, order)
        if datatype == 'daily':
            Y = np.sum(data_sums[i]) * Y / (np.sum(Y) if np.sum(Y) != 0 else 1)
            Y[Y < 0] = 0
        else:
            for n in range(len(Y) - 1):
                if Y[n + 1] - Y[n] < 0:
                    Y[n + 1] = Y[n]
            Y = np.rint(np.amax(X) * Y / np.amax(Y))
        data_filtered[:, i] = Y
        cutoff_freqs.append(cutoff_list[idx])

    return data_filtered.T, cutoff_freqs


def main():
    default_model = "['SL', 'Texas', 'NG', 'IT', 'BD', 'KZ', 'KR', 'DEU']_LSTM_Simple_WO_Regions_Filtered_Loss_50_10"
    # ============================================================================================ Initialize parameters
    parser = argparse.ArgumentParser(description='Train NN model for forecasting COVID-19 pandemic')
    parser.add_argument('--dataset', help='Dataset used for training. (Sri Lanka, Texas, USA, Global)', type=str,
                        default='SL')
    parser.add_argument('--path', help='default dataset path', type=str, default="../Datasets")
    parser.add_argument('--model', help='Model name', type=str, default=default_model)
    args = parser.parse_args()

    global DATASET, PLOT

    DATASET = args.dataset
    model_name = args.model

    midpoint = True

    if midpoint:
        R_EIG_ratio = 1.02
        R_power = 1
    else:
        R_EIG_ratio = 3
        R_power = 1
    PLOT = True
    districtwise = False
    calc_days = 20
    # ===================================================================================================== Loading data
    global to_predict, daily_filtered, population, region_names, region_mask, data_end_date

    population = None
    if districtwise:
        # if predict_deaths:
        #     raise Exception("No data to predict deaths districtwise in SL")
        df = pd.read_csv("D:\Research\COVID\sl-cov19-forecasting\Datasets\SL\SL_covid_all_updated.csv")
        df = df.dropna()
        region_names = df["District"].values
        START_DATE = df.columns[2]
    else:
        df = pd.read_csv("http://ai4covid.lk/analysisdata.csv")
        df = df.iloc[1:, :]
        region_names = ["local_new_case", "local_new_death"]
        START_DATE = df.iloc[0, 0]
    df_predict = None
    for _d in range(-calc_days, 0, 1):
        if districtwise:
            if _d != 0:
                _df = df.iloc[:, :_d]
            else:
                _df = df
            to_predict = _df.values[:, 2:]
            to_predict = to_predict[:, 1:] - to_predict[:, :-1]
            to_predict = to_predict.astype(np.float)
        else:
            if _d != 0:
                _df = df.iloc[:_d, :]
            else:
                _df = df
            to_predict = _df[region_names].values.T
        data_end_date = max(_df["Date"])
        to_predict[to_predict < 0] = 0
        n_regions = to_predict.shape[0]

        region_mask = (np.arange(n_regions) != -1).astype('int32')

        daily_filtered, cutoff_freqs = O_LPF(to_predict, datatype='daily', order=3, R_EIG_ratio=R_EIG_ratio,
                                             R_power=R_power,
                                             midpoint=midpoint,
                                             corr=True,
                                             region_names=region_names, plot_freq=1, view=False)

        x_data, y_data, x_data_scalers = get_data(False, normalize=True, data=to_predict, dataf=daily_filtered,
                                                  population=population)
        x_dataf, y_dataf, x_data_scalersf = get_data(True, normalize=True, data=to_predict, dataf=daily_filtered,
                                                     population=population)
        model_names = [(model_name, 'LSTM-F-Loss (F)'), ]
        _df_predict = get_predictions(x_data_scalers, model_names)
        if districtwise:
            _df_predict['Date'] = _df_predict['Date'].dt.strftime('%m/%d/%Y')
            _df_predict = _df_predict.set_index('Date')
            _df_predict = _df_predict.T
            _df_predict = _df_predict.join(_df[['Code', 'District']])
            cols = _df_predict.columns.tolist()
            _df_predict = _df_predict[cols[-2:] + cols[:-2]]
            _df_predict.iloc[:, 2:] = _df_predict.iloc[:, 2:].cumsum(1)
            for col in range(2, len(_df_predict.columns)):
                _df_predict.iloc[:, col] += _df[_df.columns[-1]]
                _df_predict[_df_predict.columns[col]] = _df_predict[_df_predict.columns[col]].astype(np.int)
        else:
            _df_predict = _df_predict.rename(columns={i: region_names[i] for i in range(len(region_names))})

            current_total_cases = _df["total_case"].values[-1]
            _df_predict["total_case"] = current_total_cases + _df_predict["local_new_case"].cumsum()
            current_total_deaths = _df["total_death"].values[-1]
            _df_predict["total_death"] = current_total_deaths + _df_predict["local_new_death"].cumsum()

        if df_predict is None:
            df_predict = _df_predict
        else:
            weight = 0.0
            if districtwise:
                df_predict[_df_predict.columns[-1]] = _df_predict[_df_predict.columns[-1]]
                for col in _df_predict.columns[2:]:
                    df_predict[col] = (df_predict[col] * weight + _df_predict[col] * (1 - weight))
            else:
                df_predict = df_predict.append(_df_predict.iloc[-1, :])

                for col in _df_predict.columns:
                    # if 'total' in col:
                    #     print(col)
                    #     continue
                    if col != "Date":
                        df_predict[col].iloc[-len(_df_predict):] = (
                                df_predict[col].iloc[-len(_df_predict):].values * weight +
                                _df_predict[col].values * (1 - weight))
                df_predict = df_predict[["Date", "local_new_case", "local_new_death", "total_case", "total_death"]]
        print(df_predict)
    if districtwise:
        df_filtered = _df.copy()
        df_filtered.iloc[:, 3:] = daily_filtered.cumsum(1)
        df_filtered.iloc[:, 2] = 0
        pd.DataFrame.to_csv(df_filtered, f"district_data_smooth.csv", index=False)
    else:
        df_filtered = _df.copy()
        df_filtered[region_names] = daily_filtered.T
        df_filtered[["total_case", "total_death"]] = np.nan # daily_filtered.cumsum(1).T
        pd.DataFrame.to_csv(df_filtered, f"sl_data_smooth.csv", index=False)

    if districtwise:
        # df_predict = df_predict.iloc[:,:-1]
        pd.DataFrame.to_csv(df_predict, f"district_predictions.csv", index=False)
    else:
        # df_predict = df_predict.iloc[:-1,:]
        df_predict.iloc[-2,1:]=(df_predict.iloc[-3,1:]+df_predict.iloc[-1,1:])/2
        pd.DataFrame.to_csv(df_predict, f"total_predictions.csv", index=False)
    print(df_predict)
    plt.show()

def get_ub_lb(pred, true, n_regions):
    err = abs((pred - true) ** 2)
    ub_err = np.sqrt(np.mean(err, axis=-1, keepdims=True)).repeat(n_regions, axis=-1) + pred
    lb_err = np.maximum(-np.sqrt(np.mean(err, axis=-1, keepdims=True)).repeat(n_regions, axis=-1) + pred, 0)
    return ub_err, lb_err


def get_predictions(x_data_scalers, model_names, add_raw_input=True, add_fil_input=True, add_ub_lb=False):
    print("===================================== TESTING PREDICTIONS =================================================")
    n_regions = len(x_data_scalers.data_max_)
    Ys = []
    method_list = []
    styles = {
        'X': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
        'Xf': {'Preprocessing': 'Filtered', 'Data': 'Training', 'Size': 2},
        'Observations Raw': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
        'Observations Filtered': {'Preprocessing': 'Filtered', 'Data': 'Training', 'Size': 2},
    }

    def get_model_predictions(model, x_data, scalers):
        global WINDOW_LENGTH, PREDICT_STEPS
        WINDOW_LENGTH = model.input.shape[1]
        PREDICT_STEPS = model.output.shape[1]
        X_test_w = np.array([x_data[-WINDOW_LENGTH:]])
        print(f"Predicting from model. {model.input.shape} --> {model.output.shape} X={X_test_w.shape}")

        if model.input.shape[-1] == 1:
            yhat = []
            for col in range(n_regions):
                yhat.append(model.predict(X_test_w[:, :, col:col + 1]))
            yhat = np.array(yhat)[:, :, :, 0].transpose([1, 2, 0])
        else:
            yhat = model.predict(X_test_w[:, :, :])

        # yhat[:,-1,:] =yhat[:,-2,:]+0.02

        X_test_w = undo_normalization(X_test_w, scalers)
        yhat = undo_normalization(yhat, scalers)
        return X_test_w, yhat

    x_data, y_data, _ = get_data(filtered=False, normalize=False, data=to_predict, dataf=daily_filtered,
                                 population=population, lastndays=None)
    x_dataf, y_dataf, _ = get_data(filtered=True, normalize=False, data=to_predict, dataf=daily_filtered,
                                   population=population, lastndays=None)

    #########################################################################
    for i in range(len(model_names)):
        model_filename, model_label = model_names[i]

        model = tf.keras.models.load_model(f"models/{model_filename}.h5")

        # get filtered data and predict the new cases for test period
        x_dataf, y_dataf, _ = get_data(filtered=True, normalize=x_data_scalers, data=to_predict, dataf=daily_filtered,
                                       population=population, lastndays=None)
        x_testf, yhatf = get_model_predictions(model, x_dataf, x_data_scalers)

        # get raw data and predict the new cases for test period (yhat: (days,regions))
        x_data, y_data, _ = get_data(filtered=False, normalize=x_data_scalers, data=to_predict, dataf=daily_filtered,
                                     population=population, lastndays=None)
        x_test, yhat = get_model_predictions(model, x_data, x_data_scalers)

        Ys.append(yhatf)

        method_list.append(model_label)

        styles[model_label] = {'Preprocessing': 'Filtered', 'Data': model_label, 'Size': 3}

    Ys = np.stack(Ys, 1)
    plt.figure()
    x = [pd.to_datetime(data_end_date) + pd.DateOffset(days=int(i)-x_test.shape[1]) for i in range(x_test.shape[1])]
    plt.plot(x, x_test.sum(0))
    plt.plot(x, x_testf.sum(0))
    x = [pd.to_datetime(data_end_date) + pd.DateOffset(days=int(i)) for i in range(Ys.shape[2])]
    plt.plot(x, Ys[0].sum(0), '--')

    # Ys shape - (sectors=1, models=1, predict_days, regions)
    to_save = Ys[0, 0, :, :]
    dates = [pd.to_datetime(data_end_date) + pd.DateOffset(days=int(i)) for i in range(Ys.shape[2])]

    s = pd.Series(dates, name="Date")
    df = pd.DataFrame(to_save)
    df = df.join(s)
    return df


if __name__ == "__main__":
    main()
