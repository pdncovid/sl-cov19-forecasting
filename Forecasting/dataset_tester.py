import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.signal as signal

import matplotlib.pyplot as plt

from Forecasting.utils.data_loader import load_data_eu, load_data
from Forecasting.utils.smoothing_functions import O_LPF, NO_LPF

dataset_path = 'F:\GitHub\sl-cov19-forecasting\Datasets'
_df = pd.read_csv(os.path.join(dataset_path, "EU\jrc-covid-19-all-days-by-regions.csv"))
_eu = _df['CountryName'].unique().tolist()

# %%
country = 'NG'
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
seg_len = 200
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

##%% OPTIMAL FILTERING
midpoint = False

if midpoint:
    R_weight = 2
    EIG_weight = 2
else:
    R_weight = 1
    EIG_weight = 2

daily_filtered, cutoff_freqs = O_LPF(daily_cases, datatype='daily', order=3, R_weight=R_weight,
                                     EIG_weight=EIG_weight, midpoint=midpoint, corr=True,
                                     region_names=region_names, plot_freq=1, view=True)

if midpoint:
    R_weight = 2
    EIG_weight = 2
else:
    R_weight = 3
    EIG_weight = 2

for i in range(seg_num):
    daily_seg_filtered[:, i, :], cutoff_freqs_seg = O_LPF(daily_seg[:, i, :], datatype='daily', order=3,
                                                          R_weight=R_weight,
                                                          EIG_weight=EIG_weight, midpoint=midpoint, corr=True,
                                                          region_names=region_names, plot_freq=1, view=False)

daily_filtered, cutoff_freqs = O_LPF(daily_cases, datatype='daily', order=3, R_weight=R_weight,
                                     EIG_weight=EIG_weight, midpoint=midpoint, corr=True,
                                     region_names=region_names, plot_freq=1, view=False)

daily_seg_filtered = np.reshape(daily_seg_filtered, [n_regions, -1])
daily_seg = np.reshape(daily_seg, [n_regions, -1])


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
#     f, t, Zxx = signal.stft(daily_cases[i, :], nperseg=50, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True)
#     plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
#     plt.title('STFT Magnitude')
#     plt.ylabel('Frequency [1/day]')
#     plt.xlabel('Time [day]')
#     plt.subplot(224)
#     f, t, Zxx = signal.stft(daily_filtered[i, :], nperseg=50, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True)
#     plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.amax(np.abs(Zxx)), shading='gouraud')
#     plt.title('STFT Magnitude')
#     plt.ylabel('Frequency [1/day]')
#     plt.xlabel('Time [day]')
#     plt.show()
