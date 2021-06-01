import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import signal


# ============================== Optimised LPF ========================================

def O_LPF(data, datatype, order, R_weight, EIG_weight, midpoint, corr, region_names, plot_freq, view, savepath=None):
    print(f"Smoothing {data.shape}")
    if datatype == 'daily':
        data_sums = np.zeros(data.shape[0], )
        for i in range(data.shape[0]):
            data_sums[i] = np.sum(data[i, :])

    if midpoint:
        EIG_cons = R_weight
        R_cons = EIG_weight
    else:
        R_cons = R_weight
        EIG_cons = EIG_weight

    data = np.copy(data.T)
    n_regions = data.shape[1]
    cutoff_freqs = []  # to return the optimal cutoff frequencies
    # FILTERING:
    # Filter requirements.
    T = data.shape[0]
    fs = 1
    cutoff = 0.017
    nyq = 0.5 * fs
    # order = 1
    n = int(T * fs)

    def lowpass_filter(data, cutoff, fs, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data)
        return y.astype(np.float32)

    # DETERMINE THE RIGHT CUTOFF FREQUENCY
    step = 0.01
    cutoff_list = range(int(round(1 / step)))
    cutoff_list = 0.1 * (np.array(list(cutoff_list)) + 5) / 100
    # print('cutoff_list=',cutoff_list)

    sections = 7
    if view or savepath is not None:
        cols = 5
        rows = int(np.ceil((n_regions//plot_freq)/5))
        plt.figure(89,figsize=(12*cols, 3.5*rows))

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
                Y = data_sums[i] * Y / np.sum(Y)
                # Y[Y<0]=0
                # else:
                #   for n in range(len(Y)-1):
                #     if Y[n+1]-Y[n]<0:
                #       Y[n+1]=Y[n]
                Y = np.amax(X) * Y / np.amax(Y)

            if corr:
                J_R.append(np.mean(np.corrcoef(X, Y)))  # obtaining correlations
            else:
                J_R.append(1 / np.mean(np.square(X - Y)))  # obtaining error

            # obtaining power spectral densities
            X_freqs, X_psd = signal.welch(X)
            Y_freqs, Y_psd = signal.welch(Y)

            X_psd, Y_psd = np.log10(np.abs(X_psd)), np.log10(np.abs(Y_psd))

            J0 = []

            sec_len = int(X_psd.shape[0] / sections)
            for k in range(sections):
                X_avg = np.mean(X_psd[k * sec_len:(k + 1) * sec_len])
                Y_avg = np.mean(Y_psd[k * sec_len:(k + 1) * sec_len])
                J0.append((k + 1) * np.abs(
                    X_avg - Y_avg))  # eigenvalue spread should increase as k increases for an ideal solution
            J_eig.append(np.sum(J0))
        # few assignments to get rid of errors
        J_eig = np.around(J_eig).astype(int)
        J_R = np.array(J_R)
        J_eig[J_eig < 0] = 0
        J_EIG = J_eig / (np.amax(J_eig) if np.amax(J_eig) != 0 else 0)
        J_Err = J_R / (np.amax(J_R) if np.amax(J_R) != 0 else 0)

        if midpoint:
            J_tot = 1 - np.abs(R_cons * (J_Err) - EIG_cons * (J_EIG))
        else:
            J_tot = R_cons * (J_Err) + EIG_cons * (J_EIG)

        J_tot = J_tot / np.amax(J_tot)
        idx = np.argmax(J_tot)
        Y = lowpass_filter(X, cutoff_list[idx], fs, order)
        if datatype == 'daily':
            Y = np.sum(data_sums[i]) * Y / np.sum(Y)
            Y[Y < 0] = 0
        else:
            for n in range(len(Y) - 1):
                if Y[n + 1] - Y[n] < 0:
                    Y[n + 1] = Y[n]
            Y = np.amax(X) * Y / np.amax(Y)
        data_filtered[:, i] = Y

        cutoff_freqs.append(cutoff_list[idx])

        if view:
            if i % plot_freq == 0:


                plt.subplot(rows, 2*cols, 2*i+1), plt.title('fitness functions of each component')
                plt.plot(cutoff_list, J_Err, linewidth=2)
                plt.plot(cutoff_list, J_EIG, linewidth=2)
                plt.plot(cutoff_list, J_tot, linewidth=2)
                plt.xlim([cutoff_list[0], cutoff_list[-1]])
                # plt.ylim([0,1.1])
                plt.legend(['correlation (information retained)', 'eigenvalue spread (noise removed)',
                            'total fitness function'], loc='lower left')
                plt.xlabel('normalized cutoff frequency')

                plt.subplot(rows, 2*cols, 2*i+2), plt.title(
                    'cumulative cases in ' + str(region_names[i]) + '\noptimum normalized cutoff frequency: ' + str(
                        round(cutoff_list[idx], 4)))
                plt.plot(X / np.amax(Y), linewidth=2)
                plt.plot(Y / np.amax(Y), linewidth=2, color='r')
                plt.legend(['original', 'filtered']), plt.xlabel('days')
    if view and savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')

    return data_filtered.T, cutoff_freqs


# ============================== Non-Optimised LPF ====================================

def NO_LPF(data, datatype, cutoff, order, plot=True, region_names=None):
    if datatype == 'daily':
        data_sums = np.zeros(data.shape[0], )
        for i in range(data.shape[0]):
            data_sums[i] = np.sum(data[i, :])

    data = np.copy(data.T)
    n_regions = data.shape[1]

    # FILTERING:
    # Filter requirements.
    T = data.shape[0]
    fs = 1
    nyq = 0.5 * fs
    # order = 2
    n = int(T * fs)

    def lowpass_filter(data, cutoff, fs, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data)
        return y.astype(np.float32)

    columns = 2
    rows = math.ceil(n_regions / columns)
    if plot == True:
        plt.figure(figsize=(6 * columns, 4 * rows))

    data_filtered = np.zeros_like(data)
    for i in range(n_regions):
        X = data[:, i]
        Y = lowpass_filter(X, cutoff, fs, order)
        if datatype == 'daily':
            if np.sum(Y) == 0:
                print('THERE IS AN ERROR!!!! ABORT')
            else:
                Y = np.sum(data_sums[i]) * Y / np.sum(Y)
                Y[Y < 0] = 0
        else:
            for n in range(len(Y) - 1):
                if Y[n + 1] - Y[n] < 0:
                    Y[n + 1] = Y[n]
            Y = np.amax(X) * Y / np.amax(Y)
        data_filtered[:, i] = Y

        if plot == True:
            plt.subplot(rows, columns, i + 1)
            plt.title('daily new cases in ' + str(region_names[i]))
            plt.plot(X, linewidth=2), plt.plot(Y, linewidth=2, color='r')
            plt.legend(['original', 'filtered']), plt.xlabel('days')
    if plot == True:
        plt.show()
    return data_filtered.T


# ============================== Optimised n-day avg ========================================

def O_NDA(data, region_names):
    data = np.copy(data.T)
    n_regions = data.shape[1]

    def n_day_avg(data, N):
        y = np.zeros_like(data)
        y[0] = 1 * data[0]
        for k in range(N - 1):
            y[k + 1] = np.mean(data[0:k + 1])
        for i in range(len(data) - N):
            y[i + N] = np.mean(data[i:i + N])
        return y.astype(np.float32)

    # DETERMINE THE RIGHT NO OF DAYS
    N_days = list(range(int(14)))
    N_days = np.array(N_days)
    N_days = N_days + 1
    sections = 5

    data_filtered = np.zeros_like(data)
    for i in range(n_regions):
        X_avg = []
        Y_avg = []
        J_R = []
        J_eig = []
        J_tot = []
        for n in range(len(N_days)):
            day = N_days[n]
            X = data[:, i]
            Y = n_day_avg(X, day)
            Y[Y < 0] = 0
            # obtaining power spectral densities
            X_freqs, X_psd = signal.welch(X)
            Y_freqs, Y_psd = signal.welch(Y)
            sec_len = int(X_psd.shape[0] / sections)
            X_psd, Y_psd = np.log10(np.abs(X_psd)), np.log10(np.abs(Y_psd))
            J0 = []
            for k in range(sections):
                X_avg = np.mean(X_psd[k * sec_len:(k + 1) * sec_len])
                Y_avg = np.mean(Y_psd[k * sec_len:(k + 1) * sec_len])
                J0.append((k + 1) * np.abs(
                    X_avg - Y_avg))  # eigenvalue spread should increase as k increases for an ideal solution
            J_eig.append(np.sum(J0))
            J_R.append(np.mean(np.square(X - Y)))  # obtaining correlations
        # print(J_R)
        J_tot = 3 * (1 - (J_R / np.amax(J_R))) + (J_eig / np.amax(J_eig))
        J_tot = J_tot / np.amax(J_tot)
        idx = np.argmax(J_tot)
        Y = n_day_avg(X, N_days[idx])
        Y[Y < 0] = 0
        data_filtered[:, i] = Y

        plt.figure(figsize=(12, 3.5))
        plt.subplot(1, 2, 1), plt.title('fitness functions of each component')
        plt.plot(N_days, J_R / np.amax(J_R), linewidth=2), plt.plot(N_days, J_eig / np.amax(J_eig), linewidth=2)
        plt.plot(N_days, J_tot, linewidth=2)
        plt.legend(
            ['correlation (information retained)', 'eigenvalue spread (noise removed)', 'total fitness function'],
            loc='lower left')
        plt.xlabel('days chosen for average')
        plt.subplot(1, 2, 2),
        plt.title('cumulative cases in ' + str(region_names[i]) + '\noptimum days chosen for average: ' + str(
            round(N_days[idx], 4)))
        plt.plot(X, linewidth=2), plt.plot(Y, linewidth=2, color='r')
        plt.legend(['original', 'filtered']), plt.xlabel('days')
        plt.show()
    return data_filtered.T


# ============================== Non-Optimised n-day avg ========================================

def NO_NDA(data, region_names):
    data = np.copy(data.T)
    n_regions = data.shape[1]

    day = 7

    data_filtered = np.zeros_like(data)
    for i in range(n_regions):
        X = data[:, i]
        Y = n_day_avg(X, day)
        Y[Y < 0] = 0
        # Y = np.amax(X)*Y/np.amax(Y)
        for j in range(len(Y) - 1):
            if Y[j + 1] < Y[j]:
                Y[j + 1] = Y[j]
        data_filtered[:, i] = Y

        plt.title('cumulative cases in ' + str(region_names[i]) + '\ndays chosen for average: ' + str(round(day)))
        plt.plot(X, linewidth=2), plt.plot(Y, linewidth=2, color='r')
        plt.legend(['original', 'filtered']), plt.xlabel('days')
        plt.show()
    return data_filtered.T
