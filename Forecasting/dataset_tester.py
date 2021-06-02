import os
import sys
import math
import random
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal

import matplotlib.pyplot as plt

from Forecasting.utils.data_loader import load_data_eu, load_data
from Forecasting.utils.smoothing_functions import O_LPF, NO_LPF
from Forecasting.utils.data_splitter import split_and_smooth

dataset_path = '../Datasets'
_df = pd.read_csv(os.path.join(dataset_path, "EU\jrc-covid-19-all-days-by-regions.csv"))
_eu = _df['CountryName'].unique().tolist()

# %%
country = 'Sri Lanka'
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

# %% breaking time series into segments
seg_len = 100
time_total = daily_cases.shape[1]
seg_num = math.floor(time_total / seg_len)

daily_seg = np.zeros([n_regions, seg_num, seg_len])
for k in range(n_regions):
    for i in range(seg_num):
        _idx1 = -1 - (seg_num - i) * seg_len
        _idx2 = -1 - (seg_num - 1 - i) * seg_len
        daily_seg[k, i, :] = daily_cases[k, _idx1:_idx2]

daily_seg_filtered = np.zeros_like(daily_seg)

# # %% OVER FILTERING AND UNDER FILTERING PROBLEM
# daily_filtered_na = NO_LPF(daily_cases, datatype='daily', order=10, cutoff=0.088, region_names=region_names)
# daily_filtered = NO_LPF(daily_cases, datatype='daily', order=10, cutoff=0.015, region_names=region_names)

# %% OPTIMAL FILTERING
WINDOW_LENGTH = 14
PREDICT_STEPS = 7

midpoint = False

if midpoint:
    R_weight = 2
    EIG_weight = 2
else:
    R_weight = 3
    EIG_weight = 2


daily_filtered, cutoff_freqs = O_LPF(daily_cases, datatype='daily', order=3, R_weight=R_weight,
                                     EIG_weight=EIG_weight, midpoint=midpoint, corr=True,
                                     region_names=region_names, plot_freq=1, view=True)

# daily_split_filtered, daily_split = split_and_smooth(daily_cases, look_back_window=seg_len, window_slide=10, R_weight=5,
#                                                      EIG_weight=2, midpoint=False,
#                                                      reduce_last_dim=False)


#
# daily_seg_filtered = np.reshape(daily_seg_filtered, [n_regions, -1])
# daily_seg = np.reshape(daily_seg, [n_regions, -1])
#
#
# plt.figure()
# plt.plot(daily_seg[5,:])
# plt.plot(daily_seg_filtered[5,:])
# plt.show()
# %% check FFT of each region
# for i in range(len(region_names)):
#     plt.figure(figsize=(12, 3.5))
#     plt.subplot(1, 2, 1)
#     plt.plot(daily_cases[i, :], linewidth=2)
#     plt.plot(daily_filtered[i, :], linewidth=2)
#     plt.title('filtered and unfiltered: ' + str(region_names[i]))
#     plt.subplot(1, 2, 2)
#     plt.plot(np.abs(scipy.fft.fft(daily_cases[i, :])), linewidth=2)
#     plt.plot(np.abs(scipy.fft.fft(daily_filtered[i, :])), linewidth=2)
#     plt.title('FFT: ' + str(region_names[i]))
#     plt.xlim([0, 20])
#     plt.show()


# %% STFT of each region


# for i in range(len(region_names)):
#     plt.figure(figsize=(12, 7))
#     plt.subplot(221)
#     plt.plot(daily_cases[i, :], linewidth=2)
#     plt.plot(daily_filtered[i, :], linewidth=2)
#     plt.title('filtered and unfiltered: ' + str(region_names[i]))
#     plt.subplot(222)
#     f, t, Zxx = signal.stft(daily_cases[i, :], nperseg=50, noverlap=None, nfft=None, detrend=False,
#                             return_onesided=True, boundary='zeros', padded=True)
#     plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
#     plt.title('STFT Magnitude')
#     plt.ylabel('Frequency [1/day]')
#     plt.xlabel('Time [day]')
#     plt.subplot(224)
#     f, t, Zxx = signal.stft(daily_filtered[i, :], nperseg=50, noverlap=None, nfft=None, detrend=False,
#                             return_onesided=True, boundary='zeros', padded=True)
#     plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
#     plt.title('STFT Magnitude')
#     plt.ylabel('Frequency [1/day]')
#     plt.xlabel('Time [day]')
#     plt.show()

# %% STFT of each region with splitting

# for i in range(len(region_names)):
#     # Generate 5 random numbers between 10 and 30
#     randomlist = random.sample(range(0, daily_split.shape[0]), 3)
#     # print(randomlist)
#     for k in randomlist:
#         plt.figure(figsize=(12, 7))
#         plt.subplot(221)
#         plt.plot(daily_split[k, :, i], linewidth=2)
#         plt.plot(daily_split_filtered[k, :, i], linewidth=2)
#         plt.title('filtered and unfiltered: ' + str(region_names[i]))
#         plt.subplot(222)
#         f, t, Zxx = signal.stft(daily_split[k, :, i], nperseg=20, noverlap=None, nfft=None, detrend=False,
#                                 return_onesided=True, boundary='zeros', padded=True)
#         plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
#         plt.title('STFT Magnitude')
#         plt.ylabel('Frequency [1/day]')
#         plt.xlabel('Time [day]')
#         plt.subplot(224)
#         f, t, Zxx = signal.stft(daily_split_filtered[k, :, i], nperseg=20, noverlap=None, nfft=None, detrend=False,
#                                 return_onesided=True, boundary='zeros', padded=True)
#         plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
#         plt.title('STFT Magnitude')
#         plt.ylabel('Frequency [1/day]')
#         plt.xlabel('Time [day]')
#         plt.show()

# %% under sample from whole epicurve
from utils.undersampling import undersample
x_train_uf, y_train_uf = undersample(daily_filtered, daily_filtered, WINDOW_LENGTH, PREDICT_STEPS, region_names, True)
x_train_u, y_train_u = undersample(daily_cases, daily_cases, WINDOW_LENGTH, PREDICT_STEPS, region_names, True)

# %%
plt.figure()
plt.yscale('log')
cum=False
ht='bar'
alpha = .3
s = 100
# plt.hist(daily_cases.reshape(-1), bins=np.linspace(0,daily_cases.max(),s), alpha=alpha, cumulative=cum,histtype=ht,label='Raw data')
plt.hist(daily_filtered.reshape(-1), bins=np.linspace(0,daily_cases.max(),s), alpha=alpha, cumulative=cum,histtype=ht,label='Smoothed data')
# plt.hist(x_train_u.reshape(-1), bins=np.linspace(0,daily_cases.max(),s), alpha=alpha, cumulative=cum,histtype=ht,label='Raw data (Undersampled)')
plt.hist(x_train_uf.reshape(-1), bins=np.linspace(0,daily_cases.max(),s), alpha=alpha,cumulative=cum, histtype=ht,label='Smoothed data (Undersampled)')
plt.legend()
plt.xlabel("Number of daily cases")
plt.ylabel("Frequency in the dataset")

plt.show()

#%% undersample after splitting

from utils.data_splitter import split_on_region_dimension, split_on_time_dimension, split_into_pieces_inorder, split_and_smooth
from utils.undersampling import undersample2
# %%
features = np.zeros((daily_filtered.shape[0],1))
X_train, X_train_feat, Y_train, X_val, X_val_feat, Y_val, X_test, X_test_feat, Y_test = split_on_time_dimension(
        daily_cases, daily_cases, features, WINDOW_LENGTH, PREDICT_STEPS,
        k_fold=3, test_fold=2, reduce_last_dim=False,
        only_train_test=True, debug=True)
#%%
features = np.zeros((daily_filtered.shape[0],1))
X_trainf, X_train_featf, Y_trainf, X_valf, X_val_featf, Y_valf, X_testf, X_test_featf, Y_testf = split_on_time_dimension(
        daily_filtered, daily_filtered, features, WINDOW_LENGTH, PREDICT_STEPS,
        k_fold=3, test_fold=2, reduce_last_dim=False,
        only_train_test=True, debug=True)

#%%
x_train_uf, y_train_uf = undersample2(X_trainf, Y_trainf,  region_names, True)
x_train_u, y_train_u = undersample2(X_train, Y_train,  region_names, True)


#%%
plt.figure()
# plt.yscale('log')

cum=True
ht='step'
alpha = 1
s = 100
_max = np.max([X_train.max(),X_trainf.max(),x_train_u.max(),x_train_uf.max()])
bins=np.linspace(0,_max.max(),s)

def f(x):
    return x.reshape(-1)
plt.hist(f(X_train), bins=bins, alpha=alpha, cumulative=cum,histtype=ht,label='Raw data', density=True)
plt.hist(f(X_trainf), bins=bins, alpha=alpha, cumulative=cum,histtype=ht,label='Smoothed data', density=True)
plt.hist(f(x_train_u), bins=bins, alpha=alpha, cumulative=cum,histtype=ht,label='Raw data (Undersampled)', density=True)
plt.hist(f(x_train_uf), bins=bins, alpha=alpha,cumulative=cum, histtype=ht,label='Smoothed data (Undersampled)', density=True)
plt.legend()
plt.xlabel("Number of daily cases")
plt.ylabel("Probability density in the dataset")

plt.show()