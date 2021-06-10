import os
import sys
import math
import random
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
import statsmodels.tsa.stattools as stattools

import matplotlib.pyplot as plt

from Forecasting.utils.data_loader import load_data_eu, load_data, load_train_data
from Forecasting.utils.smoothing_functions import O_LPF, NO_LPF
from Forecasting.utils.data_splitter import split_and_smooth

dataset_path = '../Datasets'
_df = pd.read_csv(os.path.join(dataset_path, "EU\jrc-covid-19-all-days-by-regions.csv"))
_eu = _df['CountryName'].unique().tolist()

plot_hist = True
undersampling = True
filtering = True
check_spectral = False
check_stft = False
check_acf = False
check_size = False

countries = ['Texas']
# countries = ['Italy', 'Sri Lanka', 'NG', 'Texas']
fft_mean = []
fft_var = []
acf_all = []
pacf_all = []


def min_max(data):
    for k in range(data.shape[0]):
        data[k, :] = (data[k, :] - np.amin(data[k, :])) / (np.amax(data[k, :]) - np.amin(data[k, :]))
    return data


for country in countries:
    print(country)
    if country in _eu:
        d = load_data_eu(country=country, path=dataset_path, provinces=True)
    else:
        d = load_data(country, path=dataset_path)

    region_names = d["region_names"]
    confirmed_cases = d["confirmed_cases"]

    daily_cases = d["daily_cases"]
    START_DATE = d["START_DATE"]
    n_regions = d["n_regions"]
    daily_cases[daily_cases < 0] = 0

    if filtering:
        # OPTIMAL FILTERING

        WINDOW_LENGTH = 14
        PREDICT_STEPS = 7

        midpoint = True

        if midpoint:
            R_EIG_ratio = 1
            R_power = 2 / 3
        else:
            R_EIG_ratio = 3
            R_power = 1

        # daily_filtered, cutoff_freqs = O_LPF(daily_cases, datatype='daily', order=5, R_EIG_ratio=R_EIG_ratio,
        #                                      R_power=R_power, midpoint=midpoint, corr=True,
        #                                      region_names=region_names, plot_freq=1, view=False)
        daily_filtered, cutoff_freqs = O_LPF(daily_cases, datatype='daily', order=3, R_EIG_ratio=1,
                                             R_power=1, midpoint=midpoint, corr=True,
                                             region_names=region_names, plot_freq=1, view=False)
        # daily_split_filtered, daily_split = split_and_smooth(daily_cases, look_back_window=100, window_slide=50,
        #                                                      R_EIG_ratio=R_EIG_ratio, R_power=R_power, order=2,
        #                                                      midpoint=False, reduce_last_dim=False, view=False)

        # for i in range(len(region_names)):
        #     plt.figure()
        #     plt.subplot(1, 2, 1)
        #     plt.plot(daily_cases[i, :])
        #     plt.plot(daily_filtered[i, :])
        #     plt.title('region: '+region_names[i]+'  cutoff: '+str(np.around(cutoff_freqs[i], 4)))
        #     plt.subplot(1, 2, 2)
        #     plt.show()

    # check FFT of each region
    if check_spectral:

        window = 20
        cases_fft = []
        for i in range(len(region_names)):
            fft_temp = np.abs(scipy.fft.fft(daily_cases[i, :]))[0:window]
            cases_fft.append(fft_temp)
        cases_fft = np.array(cases_fft)
        _mean = np.mean(cases_fft, axis=0)
        _var = np.var(cases_fft, axis=0)
        # cases_fft = cases_fft[:, 0:20]
        fft_mean.append(_mean / np.max(_mean))
        fft_var.append(_var / np.max(_var))

    if check_acf:
        window = 40
        cases_acf = []
        cases_pacf = []
        for i in range(len(region_names)):
            _acf = stattools.acf(daily_cases[i, :], adjusted=True, nlags=window, fft=True, missing="drop")
            _pacf = stattools.pacf(daily_cases[i, :], nlags=window)
            cases_acf.append(_acf)
            cases_pacf.append(_pacf)
        cases_acf = np.array(cases_acf)
        cases_pacf = np.array(cases_pacf)
        acf_all.append(np.mean(cases_acf, axis=0))
        pacf_all.append(np.mean(cases_pacf, axis=0))

    # STFT of each region

    if check_stft:
        for i in range(len(region_names)):
            plt.figure(figsize=(12, 7))
            plt.subplot(221)
            plt.plot(daily_cases[i, :], linewidth=2)
            plt.plot(daily_filtered[i, :], linewidth=2)
            plt.title('filtered and unfiltered: ' + str(region_names[i]))
            plt.subplot(222)
            f, t, Zxx = signal.stft(daily_cases[i, :], nperseg=50, noverlap=None, nfft=None, detrend=False,
                                    return_onesided=True, boundary='zeros', padded=True)
            plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [1/day]')
            plt.xlabel('Time [day]')
            plt.subplot(224)
            f, t, Zxx = signal.stft(daily_filtered[i, :], nperseg=50, noverlap=None, nfft=None, detrend=False,
                                    return_onesided=True, boundary='zeros', padded=True)
            plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [1/day]')
            plt.xlabel('Time [day]')
            plt.show()

            # STFT of each region with splitting
        for i in range(len(region_names)):
            # Generate 5 random numbers between 10 and 30
            random_list = random.sample(range(0, daily_split.shape[0]), 3)
            # print(randomlist)
            for k in random_list:
                plt.figure(figsize=(12, 7))
                plt.subplot(221)
                plt.plot(daily_split[k, :, i], linewidth=2)
                plt.plot(daily_split_filtered[k, :, i], linewidth=2)
                plt.title('filtered and unfiltered: ' + str(region_names[i]))
                plt.subplot(222)
                f, t, Zxx = signal.stft(daily_split[k, :, i], nperseg=20, noverlap=None, nfft=None, detrend=False,
                                        return_onesided=True, boundary='zeros', padded=True)
                plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
                plt.title('STFT Magnitude')
                plt.ylabel('Frequency [1/day]')
                plt.xlabel('Time [day]')
                plt.subplot(224)
                f, t, Zxx = signal.stft(daily_split_filtered[k, :, i], nperseg=20, noverlap=None, nfft=None,
                                        detrend=False,
                                        return_onesided=True, boundary='zeros', padded=True)
                plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
                plt.title('STFT Magnitude')
                plt.ylabel('Frequency [1/day]')
                plt.xlabel('Time [day]')
                plt.show()

    # under sample from whole epicurve
    if undersampling:

        count_powers = np.around(np.linspace(0.2, 2, 20), 3)
        dataset_size = daily_cases.shape[0] * daily_cases.shape[1]
        print('dataset_size = ' + str(dataset_size))
        a = (2 - 0.1) / (1000 - 25000)
        b = 2 - (a / 1000)

        count_power = np.around(dataset_size * a + b, 3)
        if count_power > 2:
            count_power = 2
        elif count_power < 0.5:
            count_power = 0.5
        print('count_power = ' + str(count_power))

        from Forecasting.utils.undersampling import undersample, undersample2, undersample3

        daily_filtered=daily_filtered[:,50:]
        n_regions, days = daily_filtered.shape

        alldata_train = daily_filtered
        samples_all = np.zeros([n_regions, days - WINDOW_LENGTH - PREDICT_STEPS, WINDOW_LENGTH + PREDICT_STEPS])
        for i in range(n_regions):
            for k in range(samples_all.shape[1]):
                samples_all[i, k, :] = alldata_train[i, k:k + WINDOW_LENGTH + PREDICT_STEPS]
        x = samples_all[:, :, :WINDOW_LENGTH].transpose([1, 2, 0])
        y = samples_all[:, :, WINDOW_LENGTH:].transpose([1, 2, 0])
        f = np.random.random((x.shape[0], 2, x.shape[2])) # dummy features
        plt.figure()
        plt.yscale('log')
        plt.subplot(121)
        plt.hist(np.concatenate(x,-1).mean(0), alpha=0.5, bins=100, label='x original')
        tmp = load_train_data("Texas",dataset_path, "Filtered", WINDOW_LENGTH, PREDICT_STEPS, True,
                              True, 50, 1)
        x, y, f, X_test, Y_test, X_test_feat, X_val, Y_val, X_val_feat = tmp
        plt.subplot(122)
        plt.hist(np.concatenate(x,-1).mean(0), alpha=0.5, bins=100,label='x loaded')
        plt.legend()
        plt.show()
        x_train_uf, y_train_uf,x_train_f = undersample3(x, y, f, count_power, region_names, True)

        # x_train_u, y_train_u, x_train_f = undersample3(daily_cases, daily_cases, count_power, WINDOW_LENGTH,
        #                                     PREDICT_STEPS, region_names, False)

        if plot_hist:
            plt.figure()
            plt.yscale('log')
            cum = False
            ht = 'bar'
            alpha = .3
            s = 100
            # plt.hist(daily_cases.reshape(-1), bins=np.linspace(0,daily_cases.max(),s), alpha=alpha, cumulative=cum,histtype=ht,label='Raw data')
            plt.hist(daily_filtered.reshape(-1), bins=np.linspace(0, daily_cases.max(), s), alpha=alpha, cumulative=cum,
                     histtype=ht, label='Smoothed data')
            # plt.hist(x_train_u.reshape(-1), bins=np.linspace(0,daily_cases.max(),s), alpha=alpha, cumulative=cum,histtype=ht,label='Raw data (Undersampled)')
            plt.hist(x_train_uf.reshape(-1), bins=np.linspace(0, daily_cases.max(), s), alpha=alpha, cumulative=cum,
                     histtype=ht,
                     label='Smoothed data (Undersampled)')
            plt.legend()
            plt.xlabel("Number of daily cases")
            plt.ylabel("Frequency in the dataset")

            plt.show()

        # # % undersample after splitting
        #
        # from Forecasting.utils.data_splitter import split_on_region_dimension, split_on_time_dimension, \
        #     split_into_pieces_inorder, \
        #     split_and_smooth
        # from Forecasting.utils.undersampling import undersample2
        #
        # # %
        # features = np.zeros((daily_filtered.shape[0], 1))
        # X_train, X_train_feat, Y_train, X_val, X_val_feat, Y_val, X_test, X_test_feat, Y_test = split_on_time_dimension(
        #     daily_cases, daily_cases, features, WINDOW_LENGTH, PREDICT_STEPS,
        #     k_fold=3, test_fold=2, reduce_last_dim=False,
        #     only_train_test=True, debug=True)
        #
        # features = np.zeros((daily_filtered.shape[0], 1))
        # X_trainf, X_train_featf, Y_trainf, X_valf, X_val_featf, Y_valf, X_testf, X_test_featf, Y_testf = split_on_time_dimension(
        #     daily_filtered, daily_filtered, features, WINDOW_LENGTH, PREDICT_STEPS,
        #     k_fold=3, test_fold=2, reduce_last_dim=False,
        #     only_train_test=True, debug=True)
        #
        # x_train_uf, y_train_uf,x_train_f = undersample2(X_trainf, Y_trainf,f, count_power, region_names, True)
        # x_train_u, y_train_u ,x_train_f= undersample2(X_train, Y_train,f, count_power, region_names, True)
        #
        # if plot_hist:
        #     plt.figure()
        #     # plt.yscale('log')
        #
        #     cum = True
        #     ht = 'step'
        #     alpha = 1
        #     s = 100
        #     _max = np.max([X_train.max(), X_trainf.max(), x_train_u.max(), x_train_uf.max()])
        #     bins = np.linspace(0, _max.max(), s)
        #
        #
        #     def f(x):
        #         return x.reshape(-1)
        #
        #
        #     plt.hist(f(X_train), bins=bins, alpha=alpha, cumulative=cum, histtype=ht, label='Raw data', density=True)
        #     plt.hist(f(X_trainf), bins=bins, alpha=alpha, cumulative=cum, histtype=ht, label='Smoothed data',
        #              density=True)
        #     plt.hist(f(x_train_u), bins=bins, alpha=alpha, cumulative=cum, histtype=ht, label='Raw data (Undersampled)',
        #              density=True)
        #     plt.hist(f(x_train_uf), bins=bins, alpha=alpha, cumulative=cum, histtype=ht,
        #              label='Smoothed data (Undersampled)',
        #              density=True)
        #     plt.legend()
        #     plt.xlabel("Number of daily cases")
        #     plt.ylabel("Probability density in the dataset")
        #
        #     plt.show()

if check_spectral:
    fft_mean = np.array(fft_mean)
    fft_var = np.array(fft_var)
    plt.figure(figsize=(12, 3.5 * len(countries)))
    for i in range(len(countries)):
        plt.subplot(len(countries), 2, 2 * i + 1)
        plt.plot(fft_mean[i, :])
        plt.title('mean fft for ' + countries[i])
        plt.subplot(len(countries), 2, 2 * i + 2)
        plt.plot(fft_var[i, :])
        plt.title('var fft for ' + countries[i])
    plt.suptitle('FFT for countries: ' + str(countries), weight='bold')
    plt.show()

if check_acf:
    acf_all = np.array(acf_all)
    pacf_all = np.array(pacf_all)
    acf_diff = np.diff(acf_all)
    pacf_diff = np.diff(pacf_all)
    plt.figure(figsize=(12 * 2, 3.5 * len(countries)))
    for i in range(len(countries)):
        plt.subplot(len(countries), 4, 4 * i + 1)
        plt.stem(acf_all[i, :])
        plt.title('mean acf for ' + countries[i])
        plt.subplot(len(countries), 4, 4 * i + 2)
        plt.stem(acf_diff[i, :]), plt.ylim([-0.3, 0.3])
        plt.title('mean acf DIFF for ' + countries[i])
        plt.subplot(len(countries), 4, 4 * i + 3)
        plt.stem(pacf_all[i, :])
        plt.title('mean pacf for ' + countries[i])
        plt.subplot(len(countries), 4, 4 * i + 4)
        plt.stem(pacf_diff[i, :]), plt.ylim([-0.3, 0.3])
        plt.title('mean pacf DIFF for ' + countries[i])
    plt.suptitle('ACF and PACF for countries: ' + str(countries), weight='bold')
    plt.show()
