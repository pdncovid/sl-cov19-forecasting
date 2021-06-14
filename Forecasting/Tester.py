#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(sys.path[0], '..'))

import pandas as pd  # Basic library for all of our dataset operations
import numpy as np
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings(
    "ignore")  # We will use deprecated models of statmodels which throw a lot of warnings to use more modern ones

from utils.metrics import evaluate
from utils.plots import bar_metrics, plot_prediction
from utils.functions import distance, normalize_for_nn, undo_normalization
from utils.data_loader import load_data, per_million, get_daily, get_data
from utils.smoothing_functions import O_LPF, NO_LPF, O_NDA, NO_NDA
from utils.data_splitter import split_on_region_dimension, split_on_time_dimension, split_into_pieces_inorder

from eval_methods.naive import naive_mean, naive_yesterday

# from eval_methods.utsf2 import SES, HWES, mAR, MA, ARIMA, SARIMA, AutoSARIMA
# from eval_methods.mtsf2 import BaysianRegression, Lasso, Randomforest, XGBoost, Lightgbm, SVM_RBF, Kneighbors


# Extra settings
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['figure.figsize'] = 18, 8

print(tf.__version__)


# x_data, y_data = get_data(filtered=False, normalize=False)
# region_mask = (np.mean(x_data,0) > 140).astype('int32')


# # Methods for time series forecasting

# There are many methods that we can use for time series forecasting and there is not a clear winner. Model selection
# should always depend on how you data look and what are you trying to achieve. Some models may be more robust against
# outliers but perform worse than the more sensible and could still be the best choice depending on the use case.
# 
# When looking at your data the main split is wether we have extra regressors (features) to our time series or just the
# series. Based on this we can start exploring different methods for forecasting and their performance in different
# metrics.
# 
# In this section we will show models for both cases, time series with and without extra regressors.

# **Prepare data before modeling**


def main():
    # ============================================================================================ Initialize parameters
    parser = argparse.ArgumentParser(description='Train NN model for forecasting COVID-19 pandemic')
    parser.add_argument('--daily', help='Use daily data', action='store_true')
    parser.add_argument('--dataset', help='Dataset used for training. (Sri Lanka, Texas, USA, Global)', type=str,
                        default='SL')
    parser.add_argument('--split_date', help='Train-Test splitting date', type=str, default='2021-2-1')

    parser.add_argument('--epochs', help='Epochs to be trained', type=int, default=10)
    parser.add_argument('--batchsize', help='Batch size', type=int, default=16)
    parser.add_argument('--input_days', help='Number of days input into the NN', type=int, default=14)
    parser.add_argument('--output_days', help='Number of days predicted by the model', type=float, default=7)
    parser.add_argument('--modeltype', help='Model type', type=str, default='LSTM_Simple_WO_Regions')

    parser.add_argument('--lr', help='Learning rate', type=int, default=0.002)
    parser.add_argument('--preprocessing', help='Preprocessing on the training data (Unfiltered, Filtered)', type=str,
                        default="Filtered")
    parser.add_argument('--undersampling', help='under-sampling method (Loss, Reduce)', type=str, default="Reduce")

    parser.add_argument('--path', help='default dataset path', type=str, default="../Datasets")
    parser.add_argument('--asymptotic_t',
                        help='Mean asymptotic period. (Test acc gradually increases with disease age)',
                        type=int, default=14)

    parser.add_argument('--initialize',
                        help='How to initialize the positions (0-Random, 1-From file 2-From probability map)', type=int,
                        default=0)

    parser.add_argument('--mobility', help='How people move around (0-Random, 1-Brownian)', type=int, default=0)
    parser.add_argument('--mobility_r', help='mobility radius', type=int, default=10)

    args = parser.parse_args()

    global daily_data, DATASET, split_date, EPOCHS, BATCH_SIZE, BUFFER_SIZE, WINDOW_LENGTH, PREDICT_STEPS, lr, TRAINING_DATA_TYPE, UNDERSAMPLING, PLOT
    daily_data = args.daily
    DATASET = args.dataset
    split_date = args.split_date

    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    BUFFER_SIZE = 100
    WINDOW_LENGTH = args.input_days
    PREDICT_STEPS = args.output_days
    lr = args.lr
    TRAINING_DATA_TYPE = args.preprocessing
    UNDERSAMPLING = args.undersampling

    midpoint = True
    R_EIG_ratio = 3
    R_power = 1
    look_back_filter = True
    look_back_window, window_slide = 50, 1
    PLOT = True

    # ===================================================================================================== Loading data
    global daily_cases, daily_filtered, population, region_names

    """Required variables:

    *   **region_names** - Names of the unique regions.
    *   **confirmed_cases** - 2D array. Each row should corresponds to values in 'region_names'. 
                            Each column represents a day. Columns should be in ascending order. 
                            (Starting day -> Present)
    *   **daily_cases** - confirmed_cases.diff()
    *   **population** - Population in 'region'
    *   **features** - Features of the regions. Each column is a certain feature.
    *   **START_DATE** - Starting date of the data DD/MM/YYYY
    *   **n_regions** Number of regions


    """

    d = load_data(DATASET, path=args.path)
    region_names = d["region_names"]
    confirmed_cases = d["confirmed_cases"]
    daily_cases = d["daily_cases"]
    features = d["features"]
    START_DATE = d["START_DATE"]
    n_regions = d["n_regions"]
    daily_cases[daily_cases < 0] = 0
    population = features["Population"]
    for i in range(len(population)):
        print("{:.2f}%".format(confirmed_cases[i, :].max() / population[i] * 100), region_names[i])

    days = confirmed_cases.shape[1]
    n_features = features.shape[1]

    global region_mask
    region_mask = (np.arange(n_regions) == 5).astype('int32')

    print(f"Total population {population.sum() / 1e6:.2f}M, regions:{n_regions}, days:{days}")

    daily_filtered, cutoff_freqs = O_LPF(daily_cases, datatype='daily', order=3, R_EIG_ratio=R_EIG_ratio,
                                         R_power=R_power,
                                         midpoint=midpoint,
                                         corr=True,
                                         region_names=region_names, plot_freq=1, view=False)

    df = pd.DataFrame(daily_cases.T, columns=features.index)
    df.index = pd.to_datetime(pd.to_datetime(START_DATE).value + df.index * 24 * 3600 * 1000000000)

    df_training = df.loc[df.index <= split_date]
    df_test = df.loc[df.index > split_date]
    print(f"{len(df_training)} days of training data \n {len(df_test)} days of testing data ")

    features = features.values
    global split_days
    split_days = (pd.to_datetime(split_date) - pd.to_datetime(START_DATE)).days

    # to_plot = ['KAL', 'GAL', 'GAM', 'HAM', 'JAF', 'KAN', 'MTL', 'MTR', 'TRI']
    to_plot = np.array(df_training.columns)
    np.random.shuffle(to_plot)
    to_plot = to_plot[:4]
    # to_plot =features.index
    if PLOT:
        plt.figure(figsize=(15, len(to_plot)))
        for i, tp in enumerate(to_plot):
            plt.subplot(1 + len(to_plot) // 3, 3, i + 1)
            plt.plot(df_training[tp], label=str(tp))
            plt.legend()

    resultsDict = {}
    predictionsDict = {}
    gtDict = {}

    mean = naive_mean(df_test)
    resultsDict['Naive mean'] = evaluate(df_test.values, mean)
    predictionsDict['Naive mean'] = mean
    gtDict['Naive mean'] = df_test.values

    yesterday = naive_yesterday(df_test)
    resultsDict['Yesterdays value'] = evaluate(df_test.values, yesterday)
    predictionsDict['Yesterdays value'] = yesterday
    gtDict['Yesterdays value'] = df_test.values

    # ses = SES(df, df_training, df_test)
    # resultsDict['SES'] = evaluate(df_test.values, ses)
    # predictionsDict['SES'] = ses
    #
    # hwes = HWES(df, df_training, df_test)
    # resultsDict['HWES'] = evaluate(df_test.values, hwes)
    # predictionsDict['HWES'] = hwes
    #
    # ar = mAR(df, df_training, df_test)
    # resultsDict['AR'] = evaluate(df_test.values, ar)
    # predictionsDict['AR'] = ar
    # # *we can observe a little delay.*
    #
    # ma = MA(df, df_training, df_test)
    # resultsDict['MA'] = evaluate(df_test.values, ma)
    # predictionsDict['MA'] = ma
    # # *this is also not fitting ne?*
    #
    # arima = ARIMA(df, df_training, df_test)
    # resultsDict['ARIMA'] = evaluate(df_test.values, arima)
    # predictionsDict['ARIMA'] = arima
    #
    # sarimax = SARIMA(df, df_training, df_test)
    # resultsDict['SARIMAX'] = evaluate(df_test.values, sarimax)
    # predictionsDict['SARIMAX'] = sarimax
    #
    # autosarimax = AutoSARIMA(df, df_training, df_test)
    # resultsDict['AutoSARIMAX'] = evaluate(df_test.values, autosarimax)
    # predictionsDict['AutoSARIMAX'] = autosarimax

    # br = BaysianRegression(df, df_training, df_test)
    # resultsDict['BayesianRidge'] = evaluate(df_test.values, br)
    # predictionsDict['BayesianRidge'] = br
    #
    # lasso = Lasso(df, df_training, df_test)
    # resultsDict['Lasso'] = evaluate(df_test.values, lasso)
    # predictionsDict['Lasso'] = lasso
    #
    # rf = Randomforest(df, df_training, df_test)
    # resultsDict['Randomforest'] = evaluate(df_test.values, rf)
    # predictionsDict['Randomforest'] = rf
    #
    # xg = XGBoost(df, df_training, df_test)
    # resultsDict['XGBoost'] = evaluate(df_test.values, xg)
    # predictionsDict['XGBoost'] = xg
    #
    # lgbm = Lightgbm(df, df_training, df_test)
    # resultsDict['Lightgbm'] = evaluate(df_test.values, lgbm)
    # predictionsDict['Lightgbm'] = lgbm
    #
    # svmrbf = SVM_RBF(df, df_training, df_test)
    # resultsDict['SVM RBF'] = evaluate(df_test.values, svmrbf)
    # predictionsDict['SVM RBF'] = svmrbf
    #
    # kn = Kneighbors(df, df_training, df_test)
    # resultsDict['Kneighbors'] = evaluate(df_test.values, kn)
    # predictionsDict['Kneighbors'] = kn

    for method in predictionsDict.keys():
        if method not in gtDict.keys():
            gtDict[method] = df_test.values
    if PLOT:
        for method in predictionsDict.keys():
            plt.figure(figsize=(15, len(to_plot)))
            plt.title(method)
            yhat = predictionsDict[method]
            for i, tp in enumerate(to_plot):
                plt.subplot(1 + len(to_plot) // 3, 3, i + 1)
                plt.plot(df_test[tp].values, label='Original ' + str(tp))
                plt.plot(yhat[:, list(df_test.columns).index(tp)], color='red', label=method + ' ' + str(tp))
                plt.legend()
            plt.show()

    # ================================================================================================### Deep learning
    x_data, y_data, x_data_scalers = get_data(False, normalize=True, data=daily_cases, dataf=daily_filtered,
                                              population=population)
    x_dataf, y_dataf, x_data_scalersf = get_data(True, normalize=True, data=daily_cases, dataf=daily_filtered,
                                                 population=population)

    # model_names = [('SL_LSTM_Simple_WO_Regions_Filtered_Reduce_14_7', 'LSTM-F-Under'),
    #                ('SL_LSTM_Simple_WO_Regions_Unfiltered_Reduce_14_7', 'LSTM-R-Under'), ]
    # plot_data = [[{'label_name': 'Method A', 'line_size': 4}, {}],
    #              [{'label_name': 'Method C', 'line_size': 4}, {'label_name': 'Method B', 'line_size': 3}],]

    # model_names = [
    #     ('SL_LSTM_Simple_WO_Regions_Unfiltered_Loss_14_7', 'LSTM-R-Loss'),
    #     ('SL_LSTM_Simple_WO_Regions_Filtered_Loss_14_7', 'LSTM-F-Loss'),
    # ]
    # plot_data = [[{'label_name': model_names[0][1] + '-raw', 'line_size': 4},
    #               {}],
    #              [{'label_name': model_names[1][1] + '-raw', 'line_size': 4},
    #               {'label_name': model_names[1][1] + '-fil', 'line_size': 3}],
    #              ]

    # country = args.dataset
    # modeltype = 'LSTM_Simple_WO_Regions'
    # model_names = [
    #     (f'{country}_{modeltype}_Unfiltered_None_14_7', 'LSTM-R-None'),
    #     (f'{country}_{modeltype}_Filtered_None_14_7', 'LSTM-F-None'),
    #     # ('SL_{modeltype}_Unfiltered_Loss_14_7', 'LSTM-R-Loss'),
    #     # ('SL_{modeltype}_Filtered_Loss_14_7', 'LSTM-F-Loss'),
    #     (f'{country}_{modeltype}_Unfiltered_Reduce_14_7', 'LSTM-R-Reduce'),
    #     (f'{country}_{modeltype}_Filtered_Reduce_14_7', 'LSTM-F-Reduce'),
    # ]
    # plot_data = [[{'label_name': model_names[0][1] + '-raw', 'line_size': 4}, {}],
    #              [{}, {'label_name': model_names[1][1] + '-fil', 'line_size': 3}],
    #              [{'label_name': model_names[2][1] + '-raw', 'line_size': 4}, {}],
    #              [{}, {'label_name': model_names[3][1] + '-fil', 'line_size': 3}],
    #              # [{'label_name': model_names[4][1] + '-raw', 'line_size': 4}, {}],
    #              # [{}, {'label_name': model_names[5][1] + '-fil', 'line_size': 3}],
    #              ]
    model_names = [
        ("['SL', 'Texas', 'NG', 'IT']_LSTM_Simple_WO_Regions_Filtered_Reduce_5_10", 'LSTM-ALL-F-Reduce-5'),
        ("['SL', 'Texas', 'NG', 'IT']_LSTM_Simple_WO_Regions_Filtered_Reduce_15_10", 'LSTM-ALL-F-Reduce-15'),
        ("['SL', 'Texas', 'NG', 'IT']_LSTM_Simple_WO_Regions_Filtered_Reduce_25_10", 'LSTM-ALL-F-Reduce-25'),
        ("['SL', 'Texas', 'NG', 'IT']_LSTM_Simple_WO_Regions_Filtered_Reduce_50_10", 'LSTM-ALL-F-Reduce-50'),
    ]
    plot_data = [[{},  # {'label_name': model_names[0][1] + '-raw', 'line_size': 4},
                  {'label_name': model_names[0][1] + '-fil', 'line_size': 3}],
                 [{},  # {'label_name': model_names[1][1] + '-raw', 'line_size': 4},
                  {'label_name': model_names[1][1] + '-fil', 'line_size': 3}],
                 [{},  # {'label_name': model_names[2][1] + '-raw', 'line_size': 4},
                  {'label_name': model_names[2][1] + '-fil', 'line_size': 3}],
                 [{},  # {'label_name': model_names[3][1] + '-raw', 'line_size': 4},
                  {'label_name': model_names[3][1] + '-fil', 'line_size': 3}]
                 ]

    show_predictions2(x_data_scalers, resultsDict, predictionsDict, gtDict, model_names, plot_data, use_f_gt=False)

    show_pred_daybyday(x_data_scalers, resultsDict, predictionsDict, gtDict, model_names, plot_data, use_f_gt=True)
    show_pred_evolution(x_data_scalers, resultsDict, predictionsDict, gtDict, model_names, plot_data, use_f_gt=True)

    # ======================================================================================== ## Comparison of methods

    plt.figure(figsize=(15, 8))
    arr = []
    i = 0
    for method in resultsDict.keys():
        # if method == "Yesterdays value":
        #     continue
        err = predictionsDict[method] - gtDict[method]
        abserr = np.abs(err)
        sqderr = err ** 2
        mape = (abserr / (abs(gtDict[method]) + abs(predictionsDict[method]) + 1e-5) * 100)
        # mean = 0
        # for r in range(len(resultsDict[method])):
        #     mean += resultsDict[method][r][metric]
        # mean = mean/len(resultsDict[method])
        # arr.append(mean)

        n, bins, patches = plt.hist(abserr.reshape(-1), 1000, density=True, histtype='step',
                                    cumulative=True, label=method)
        i += 1

        patches[0].set_xy(patches[0].get_xy()[:-1])

        print(f'{method}\t{np.mean(abserr):.2f}\t{np.mean(sqderr) ** 0.5:.2f}\t{np.mean(mape):.2f}')

    plt.legend(loc='lower right')
    plt.xlabel("Absolute error")
    plt.ylabel("Cumulative probability density")

    plt.plot(np.mean(arr, -1).T)
    plt.legend(predictionsDict.keys())
    plt.show()

    # import pickle
    #
    # with open('results/scores.pickle', 'wb') as handle:
    #     pickle.dump(resultsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('results/predictions.pickle', 'wb') as handle:
    #     pickle.dump(predictionsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_ub_lb(pred, true, n_regions):
    err = abs((pred - true) ** 2)
    ub_err = np.sqrt(np.mean(err, axis=-1, keepdims=True)).repeat(n_regions, axis=-1) + pred
    lb_err = -np.sqrt(np.mean(err, axis=-1, keepdims=True)).repeat(n_regions, axis=-1) + pred
    return ub_err, lb_err


def show_predictions2(x_data_scalers, resultsDict, predictionsDict, gtDict, model_names, plot_data, use_f_gt=True):
    n_regions = len(x_data_scalers.data_max_)
    Ys = []
    method_list = []
    styles = {
        'X': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
        'Xf': {'Preprocessing': 'Filtered', 'Data': 'Training', 'Size': 2},
        'Observations Raw': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
        'Observations Filtered': {'Preprocessing': 'Filtered', 'Data': 'Training', 'Size': 2},
    }

    def window_data(X, Y, window=14, pred=7):
        '''
        The dataset length will be reduced to guarante all samples have the window, so new length will be len(dataset)-window
        '''
        x = []
        y = []
        for i in range(window, len(X) - pred + 1):
            x.append(X[i - window:i])
            y.append(Y[i:i + pred])
        return np.array(x), np.array(y)

    def get_model_predictions(model, x_data, y_data, scalers):
        WINDOW_LENGTH = model.input.shape[1]
        PREDICT_STEPS = model.output.shape[1]
        print(model.input.shape, "-->", model.output.shape)
        print(f"Predicting from model. X={x_data.shape} Y={y_data.shape}")
        X_test_w, y_test_w = window_data(x_data, y_data, window=WINDOW_LENGTH, pred=PREDICT_STEPS)

        if model.input.shape[-1] == 1:
            yhat = []
            for col in range(n_regions):
                yhat.append(model.predict(X_test_w[:, :, col:col + 1]))
            yhat = np.squeeze(np.array(yhat)).transpose([1, 2, 0])
        else:
            yhat = model.predict(X_test_w[:, :, :])

        X_test_w = undo_normalization(X_test_w, scalers)
        yhat = undo_normalization(yhat, scalers)
        y_test_w = undo_normalization(y_test_w, scalers)
        return X_test_w, y_test_w, yhat

    x_data, y_data, _ = get_data(filtered=False, normalize=False, data=daily_cases, dataf=daily_filtered,
                                 population=population)
    x_dataf, y_dataf, _ = get_data(filtered=True, normalize=False, data=daily_cases, dataf=daily_filtered,
                                   population=population)

    #########################################################################
    for i in range(len(model_names)):
        model_filename, model_label = model_names[i]
        plot = plot_data[i]
        model = tf.keras.models.load_model(f"models/{model_filename}.h5")

        # get filtered data and predict the new cases for test period
        x_dataf, y_dataf, _ = get_data(filtered=True, normalize=x_data_scalers, data=daily_cases, dataf=daily_filtered,
                                       population=population)
        x_testf, y_testf, yhatf = get_model_predictions(model, x_dataf, y_dataf, x_data_scalers)

        # get raw data and predict the new cases for test period (yhat: (days,regions))
        x_data, y_data, _ = get_data(filtered=False, normalize=x_data_scalers, data=daily_cases, dataf=daily_filtered,
                                     population=population)
        x_test, y_test, yhat = get_model_predictions(model, x_data, y_data, x_data_scalers)
        ygt = y_testf if use_f_gt else y_test
        if len(plot[0].keys()) != 0:
            resultsDict[f'{model_label} (R)'] = evaluate(ygt, yhat)  # raw predictions v raw true values
            predictionsDict[f'{model_label} (R)'] = yhat
            gtDict[f'{model_label} (R)'] = ygt

            Ys.append(yhat)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}

            # upper bound and lower bound
            ub_err, lb_err = get_ub_lb(predictionsDict[f'{model_label} (R)'], gtDict[f'{model_label} (R)'],
                                       n_regions)

            Ys.append(ub_err)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}

            Ys.append(lb_err)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}

        if len(plot[1].keys()) != 0:
            resultsDict[f'{model_label} (F)'] = evaluate(ygt, yhatf)  # filtered prediction v raw true values
            predictionsDict[f'{model_label} (F)'] = yhatf
            gtDict[f'{model_label} (F)'] = ygt

            Ys.append(yhatf)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}

            ub_err, lb_err = get_ub_lb(predictionsDict[f'{model_label} (F)'],
                                       gtDict[f'{model_label} (F)'], n_regions)
            Ys.append(ub_err)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}
            Ys.append(lb_err)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}

    #########################################################################
    Ys = [y_test, y_testf] + Ys
    method_list = ['Observations Raw', 'Observations Filtered'] + method_list
    _cut = 1e10
    for i in range(len(Ys)):
        _cut = min(Ys[i].shape[0], _cut)
    for i in range(len(Ys)):
        print(method_list[i], Ys[i].shape, np.arange(0, Ys[i].shape[0], WINDOW_LENGTH + PREDICT_STEPS))
        Ys[i] = Ys[i][-_cut:, :, :]
        Ys[i] = Ys[i][np.arange(0, Ys[i].shape[0], WINDOW_LENGTH + PREDICT_STEPS), :, :]
    Ys = np.stack(Ys, 1)

    x_test = x_test[np.arange(0, x_test.shape[0], WINDOW_LENGTH + PREDICT_STEPS), -WINDOW_LENGTH:, :]
    x_testf = x_testf[np.arange(0, x_testf.shape[0], WINDOW_LENGTH + PREDICT_STEPS), -WINDOW_LENGTH:, :]
    plt.figure(figsize=(18, 9))

    plot_prediction(x_test, x_testf, Ys, method_list, styles, region_names, region_mask)

    # plt.savefig(f"images/{DATASET}_DayByDay.eps")
    # plt.savefig(f"images/{DATASET}_DayByDay.jpg")
    plt.show()


# ###==================================================== Continuous prediction into future from given sequence of data.


# #### Prediction of next day from last 14 days for the test period
def show_pred_daybyday(x_data_scalers, resultsDict, predictionsDict, gtDict, model_names, plot_data, use_f_gt=True):
    """
    model_names : list of tuples [(model names to load, model label), ...]  let SIZE = n
    plot_data : list of dictionaries [ [ {dict for raw pred} , {dict for filtered pred} ], ... ] SIZE == n
                if not plotting empty dict
                otherwise dict should contain; label_name, line_size
    """

    n_regions = len(x_data_scalers.data_max_)

    def window_data(X, Y, window=7):
        '''
        The dataset length will be reduced to guarante all samples have the window, so new length will be len(dataset)-window
        '''
        x = []
        y = []
        for i in range(window - 1, len(X)):
            x.append(X[i - window + 1:i + 1])
            y.append(Y[i])
        return np.array(x), np.array(y)

    def get_model_predictions(model, x_data, y_data, scalers):
        WINDOW_LENGTH = model.input.shape[1]
        PREDICT_STEPS = model.output.shape[1]
        X_w, y_w = window_data(x_data, y_data, window=WINDOW_LENGTH)
        if split_days - WINDOW_LENGTH - 1 < 0:
            raise Exception(
                f"Test data too small to  predict  from all models. Try to increase test data size! {split_days} <= {WINDOW_LENGTH}")
        X_test_w = X_w[split_days - WINDOW_LENGTH - 1:-1]
        y_test_w = y_w[split_days - WINDOW_LENGTH - 1:-1]

        print(f"Predicting from model. X={X_test_w.shape} Y={y_test_w.shape}")

        if model.input.shape[-1] == 1:
            yhat = []
            for col in range(n_regions):
                yhat.append(model.predict(X_test_w[:, :, col:col + 1])[:, 0].reshape(1, -1)[0])
            yhat = np.squeeze(np.array(yhat)).T
        else:
            yhat = model.predict(X_test_w[:, :, :])[:, 0].reshape(-1, n_regions)

        yhat = undo_normalization(yhat, scalers)[0]
        y_test_w = undo_normalization(y_test_w, scalers)[0]
        return X_test_w, y_test_w, yhat

    x_data, y_data, _ = get_data(filtered=False, normalize=False, data=daily_cases, dataf=daily_filtered,
                                 population=population)
    x_dataf, y_dataf, _ = get_data(filtered=True, normalize=False, data=daily_cases, dataf=daily_filtered,
                                   population=population)
    X = np.expand_dims(x_data[split_days - 14:split_days, :], 0)
    Xf = np.expand_dims(x_dataf[split_days - 14:split_days, :], 0)
    # X = np.expand_dims(x_data[:split_days,:],0)
    # Xf = np.expand_dims(x_dataf[:split_days,:],0)
    Y = y_data[split_days - 1:, :]
    Yf = y_dataf[split_days - 1:, :]

    Ys = [Y]
    method_list = ['Observations Raw']
    styles = {
        'X': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
        'Xf': {'Preprocessing': 'Filtered', 'Data': 'Training', 'Size': 2},
        'Observations Raw': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
    }

    #########################################################################
    for i in range(len(model_names)):
        model_filename, model_label = model_names[i]
        plot = plot_data[i]
        model = tf.keras.models.load_model(f"models/{model_filename}.h5")

        # get filtered data and predict the new cases for test period
        x_dataf, y_dataf, _ = get_data(filtered=True, normalize=x_data_scalers, data=daily_cases, dataf=daily_filtered,
                                       population=population)
        _, y_testf, yhatf = get_model_predictions(model, x_dataf, y_dataf, x_data_scalers)

        # get raw data and predict the new cases for test period (yhat: (days,regions))
        x_data, y_data, _ = get_data(filtered=False, normalize=x_data_scalers, data=daily_cases, dataf=daily_filtered,
                                     population=population)
        _, y_test, yhat = get_model_predictions(model, x_data, y_data, x_data_scalers)
        ygt = y_testf if use_f_gt else y_test
        if len(plot[0].keys()) != 0:
            resultsDict[f'{model_label} (R-D)'] = evaluate(ygt, yhat)  # raw predictions v raw true values
            predictionsDict[f'{model_label} (R-D)'] = yhat
            gtDict[f'{model_label} (R-D)'] = ygt

            Ys.append(yhat)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}

            # upper bound and lower bound
            ub_err, lb_err = get_ub_lb(predictionsDict[f'{model_label} (R-D)'], gtDict[f'{model_label} (R-D)'],
                                       n_regions)

            Ys.append(ub_err)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}

            Ys.append(lb_err)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}

        if len(plot[1].keys()) != 0:
            resultsDict[f'{model_label} (F-D)'] = evaluate(ygt, yhatf)  # filtered prediction v raw true values
            predictionsDict[f'{model_label} (F-D)'] = yhatf
            gtDict[f'{model_label} (F-D)'] = ygt

            Ys.append(yhatf)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}

            ub_err, lb_err = get_ub_lb(predictionsDict[f'{model_label} (F-D)'],
                                       gtDict[f'{model_label} (F-D)'], n_regions)
            Ys.append(ub_err)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}
            Ys.append(lb_err)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}

    #########################################################################

    for i in range(len(Ys)):
        print(method_list[i], Ys[i].shape)
        Ys[i] = np.expand_dims(Ys[i], 0)
    Ys = np.stack(Ys, 1)

    plt.figure(figsize=(18, 9))

    plot_prediction(X, Xf, Ys, method_list, styles, region_names, region_mask)

    # plt.savefig(f"images/{DATASET}_DayByDay.eps")
    # plt.savefig(f"images/{DATASET}_DayByDay.jpg")
    plt.show()


# #### Model prediction evolution from given only last 14 days of data.
def show_pred_evolution(x_data_scalers, resultsDict, predictionsDict, gtDict, model_names, plot_data, use_f_gt=True):
    def get_model_predictions(model, x_data, y_data, scalers):
        WINDOW_LENGTH = model.input.shape[1]
        PREDICT_STEPS = model.output.shape[1]
        print(f"Predicting from model. X={x_data.shape} Y={y_data.shape}")
        X_test_w = x_data[split_days - WINDOW_LENGTH - 1:split_days - 1, :]
        y_test_w = y_data[split_days - 1:, :]

        print(X_test_w.shape, y_test_w.shape)

        if model.input.shape[-1] == 1:
            X_test_w = np.expand_dims(X_test_w.T, -1)  # shape = regions (samples), window size, 1

            yhat = []
            for day in range(split_days - 1, x_data.shape[0]):
                y_pred = model.predict(X_test_w)

                X_test_w[:, :-1, :] = X_test_w[:, 1:, :]
                X_test_w[:, -1, :] = y_pred[:, 0:1, 0]

                yhat.append(y_pred[:, 0])

        else:
            X_test_w = np.expand_dims(X_test_w, 0)  # shape = 1, window size, regions (samples)
            yhat = []
            for day in range(split_days - 1, x_data.shape[0]):
                y_pred = model.predict(X_test_w)

                X_test_w[:, :-1, :] = X_test_w[:, 1:, :]
                X_test_w[:, -1, :] = y_pred[:, 0:1, :]

                yhat.append(y_pred[:, 0])

        yhat = np.squeeze(np.array(yhat))
        print(yhat.shape, y_test_w.shape)

        yhat = undo_normalization(yhat, scalers)[0]
        y_test_w = undo_normalization(y_test_w, scalers)[0]

        return X_test_w, y_test_w, yhat

    n_regions = len(x_data_scalers.data_max_)
    x_data, y_data, _ = get_data(filtered=False, normalize=False, data=daily_cases, dataf=daily_filtered,
                                 population=population)
    x_dataf, y_dataf, _ = get_data(filtered=True, normalize=False, data=daily_cases, dataf=daily_filtered,
                                   population=population)
    X = np.expand_dims(x_data[split_days - 14:split_days, :], 0)
    Xf = np.expand_dims(x_dataf[split_days - 14:split_days, :], 0)
    # X = np.expand_dims(x_data[:split_days,:],0)
    # Xf = np.expand_dims(x_dataf[:split_days,:],0)
    Y = y_data[split_days - 1:, :]
    Yf = y_dataf[split_days - 1:, :]

    Ys = [Y]
    method_list = ['Observations Raw']
    styles = {
        'X': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
        'Xf': {'Preprocessing': 'Filtered', 'Data': 'Training', 'Size': 2},
        'Observations Raw': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
    }

    #########################################################################
    for i in range(len(model_names)):
        model_filename, model_label = model_names[i]
        plot = plot_data[i]
        model = tf.keras.models.load_model(f"models/{model_filename}.h5")
        x_dataf, y_dataf, _ = get_data(filtered=True, normalize=x_data_scalers, data=daily_cases, dataf=daily_filtered,
                                       population=population)
        _, y_testf, yhatf = get_model_predictions(model, x_dataf, y_dataf, x_data_scalers)

        x_data, y_data, _ = get_data(filtered=False, normalize=x_data_scalers, data=daily_cases, dataf=daily_filtered,
                                     population=population)
        _, y_test, yhat = get_model_predictions(model, x_data, y_data, x_data_scalers)
        ygt = y_testf if use_f_gt else y_test
        if len(plot[0].keys()) != 0:
            resultsDict[f'{model_label} (R-E)'] = evaluate(ygt, yhat)  # raw predictions v raw true values
            predictionsDict[f'{model_label} (R-E)'] = yhat
            gtDict[f'{model_label} (R-E)'] = ygt

            Ys.append(yhat)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}
            # upper bound and lower bound
            ub_err, lb_err = get_ub_lb(predictionsDict[f'{model_label} (R-E)'], gtDict[f'{model_label} (R-E)'],
                                       n_regions)
            Ys.append(ub_err)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}
            Ys.append(lb_err)
            method_name = plot[0]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Raw', 'Data': method_name, 'Size': plot[0]['line_size']}

        if len(plot[1].keys()) != 0:
            resultsDict[f'{model_label} (F-E)'] = evaluate(ygt, yhatf)  # filtered prediction v raw true values
            predictionsDict[f'{model_label} (F-E)'] = yhatf
            gtDict[f'{model_label} (F-E)'] = ygt

            Ys.append(yhatf)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}
            # upper bound and lower bound
            ub_err, lb_err = get_ub_lb(predictionsDict[f'{model_label} (F-E)'],
                                       gtDict[f'{model_label} (F-E)'], n_regions)
            Ys.append(ub_err)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}
            Ys.append(lb_err)
            method_name = plot[1]['label_name']
            method_list.append(method_name)
            styles[method_name] = {'Preprocessing': 'Filtered', 'Data': method_name, 'Size': plot[1]['line_size']}
    #########################################################################

    for i in range(len(Ys)):
        print(method_list[i], Ys[i].shape)
        Ys[i] = np.expand_dims(Ys[i], 0)
    Ys = np.stack(Ys, 1)

    plt.figure(figsize=(18, 9))
    plot_prediction(X, Xf, Ys, method_list, styles, region_names, region_mask)

    # plt.savefig(f"images/{DATASET}_Evolution.eps")
    # plt.savefig(f"images/{DATASET}_Evolution.jpg")
    plt.show()


def test_evolution(model):
    x = model.input.shape[-2]
    r = model.input.shape[-1]
    start_seqs = [np.random.random((1, x, r)),
                  np.ones((1, x, r)) * 0,
                  np.ones((1, x, r)) * 0.5,
                  np.ones((1, x, r)) * 1,
                  np.arange(x * r).reshape((1, x, r)) / 30,
                  np.sin(np.arange(x) / x * np.pi / 2).reshape((1, x, 1)).repeat(r, -1)
                  ]

    model = tf.keras.models.load_model("models/Sri Lanka_LSTM_Filtered.h5")
    predictions = []
    for start_seq in start_seqs:
        input_seq = np.copy(start_seq)
        print(input_seq.shape)
        predict_seq = [start_seq[0, :, :]]
        for _ in range(50):
            output = model(input_seq, training=False)

            input_seq = input_seq[:, output.shape[1]:, :]
            if len(output.shape) == 2:
                output = np.expand_dims(output, -1)
            predict_seq.append(output[0])
            input_seq = np.concatenate([input_seq, output], 1)
        predictions.append(np.concatenate(predict_seq, 0))

    plt.semilogy(1 + np.array(predictions)[:, :30, 0].T)
    plt.title("Model trained using filtered data")
    plt.show()

    model = tf.keras.models.load_model("models/Sri Lanka_LSTM_Unfiltered.h5")
    predictions = []
    for start_seq in start_seqs:
        input_seq = np.copy(start_seq)
        print(input_seq.shape)
        predict_seq = [start_seq[0, :, :]]
        for _ in range(50):
            output = model(input_seq, training=False)

            input_seq = input_seq[:, output.shape[1]:, :]
            if len(output.shape) == 2:
                output = np.expand_dims(output, -1)
            predict_seq.append(output[0])
            input_seq = np.concatenate([input_seq, output], 1)
        predictions.append(np.concatenate(predict_seq, 0))

    plt.semilogy(1 + np.array(predictions)[:, :30, 0].T)
    plt.title("Model trained using unfiltered data")
    plt.show()


if __name__ == "__main__":
    main()
