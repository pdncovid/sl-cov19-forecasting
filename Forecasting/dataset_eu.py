import os
import sys
import numpy as np
import scipy

import matplotlib.pyplot as plt

from Forecasting.utils.data_loader import load_data_eu, load_data
from Forecasting.utils.smoothing_functions import O_LPF, NO_LPF

dataset_path = 'F:\GitHub\sl-cov19-forecasting\Datasets'

country = 'Italy'
d = load_data_eu(country=country, path=dataset_path, provinces=True)
# d = load_data("NG", path=dataset_path)

region_names = d["region_names"]
confirmed_cases = d["confirmed_cases"]

daily_cases = d["daily_cases"]
START_DATE = d["START_DATE"]
n_regions = d["n_regions"]
daily_cases[daily_cases < 0] = 0


# #%%
# plt.figure()
# plt.plot(daily_cases[1, :])
# plt.show()
# # %% OVER FILTERING AND UNDER FILTERING PROBLEM
# daily_filtered_na = NO_LPF(daily_cases, datatype='daily', order=10, cutoff=0.088, region_names=region_names)
# daily_filtered = NO_LPF(daily_cases, datatype='daily', order=10, cutoff=0.015, region_names=region_names)

# %% OPTIMAL FILTERING
daily_filtered = O_LPF(daily_cases, datatype='daily', order=4, R_weight=1, EIG_weight=1, midpoint=False, corr=True,
                       region_names=region_names, plot_freq=1, view=True)

# %%
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
