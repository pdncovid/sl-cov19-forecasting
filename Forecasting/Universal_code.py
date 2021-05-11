#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
import time

# machine learning
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

# data manipulation and signal processing
import math
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import scipy.stats as ss

# plots
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import folium
import pydot

path = "F:\GitHub\sl-cov19-forecasting"
# os.chdir(path)
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from Forecasting.utils.plots import bar_metrics, plot_prediction
from Forecasting.utils.functions import split_into_pieces_inorder, split_into_pieces_random, create_dataset_random, \
    distance
from Forecasting.utils.functions import convert_lon_lat_to_adjacency_matrix
from Forecasting.utils.data_loader import load_data, per_million, get_daily
from Forecasting.utils.data_splitter import split_on_region_dimension, split_on_time_dimension
from Forecasting.utils.smoothing_functions import O_LPF, NO_LPF, O_NDA, NO_NDA

# # EXTRACTING DATA

# In[ ]:


daily_data = True
DATASET = "Sri Lanka"  # "Texas" "USA" "Global"
# DATASET = "Texas"


# ## Loading data

# Required variables:
# 
# *   **region_names** - Names of the unique regions.
# *   **confirmed_cases** - 2D array. Each row should corresponds to values in 'region_names'. Each column represents a day. Columns should be in ascending order. (Starting day -> Present)
# *   **daily_cases** - confirmed_cases.diff()
# *   **population** - Population in 'region'
# *   **features** - Features of the regions. Each column is a certain feature.
# *   **START_DATE** - Starting date of the data DD/MM/YYYY
# *   **n_regions** Number of regions
# 
# 

# In[ ]:


d = load_data(DATASET, path="../Datasets")
region_names = d["region_names"]
confirmed_cases = d["confirmed_cases"]
daily_cases = d["daily_cases"]
features = d["features"]
START_DATE = d["START_DATE"]
n_regions = d["n_regions"]

population = features["Population"]
for i in range(len(population)):
    print("{:.2f}%".format(confirmed_cases[i, :].max() / population[i] * 100), region_names[i])

days = confirmed_cases.shape[1]

print(f"Total population {population.sum() / 1e6:.2f}M, regions:{n_regions}, days:{days}")


# # # Checking the overfiltering and underfiltering problem (approx LINE 85 TO 222)
# # In[ ]:
# def NO_LPF(data, datatype, cutoff, order,plot=True,region_names=None):
#
#     if datatype == 'daily':
#         data_sums = np.zeros(data.shape[0],)
#         for i in range(data.shape[0]):
#             data_sums[i] = np.sum(data[i,:])
#
#     data = np.copy(data.T)
#     n_regions = data.shape[1]
#
#     # FILTERING:
#     # Filter requirements.
#     T = data.shape[0]
#     fs = 1
#     nyq = 0.5 * fs
#     # order = 2
#     n = int(T * fs)
#
#     def lowpass_filter(data, cutoff, fs, order):
#         normal_cutoff = cutoff / nyq
#       # Get the filter coefficients
#         b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
#         y = signal.filtfilt(b, a, data)
#         return y.astype(np.float32)
#
#
#     columns = 2
#     rows = math.ceil(n_regions/columns)
#     if plot==True:
#         plt.figure(figsize=(6*columns, 4*rows))
#
#     data_filtered = np.zeros_like(data)
#     for i in range(n_regions):
#         X = data[:,i]
#         X_temp = X/np.amax(X)
#         Y_temp = lowpass_filter(X_temp, cutoff, fs, order)
#         if datatype == 'daily':
#               # Y = np.sum(data_sums[i])*Y_temp/np.sum(Y_temp)
#             Y = np.copy(Y_temp)
#             Y[Y<0]=0
#         else:
#             for n in range(len(Y)-1):
#                 if Y[n+1]-Y[n]<0:
#                     Y[n+1]=Y[n]
#                     Y = np.amax(X)*Y/np.amax(Y)
#         data_filtered[:,i] = Y
#
#         if plot==True:
#             plt.subplot(rows,columns,i+1)
#             plt.title('daily new cases in '+str(region_names[i]))
#             plt.plot(X,linewidth=2),plt.plot(Y,linewidth=2,color='r')
#             plt.legend(['original','filtered']),plt.xlabel('days')
#     if plot==True:
#         plt.show()
#     return data_filtered.T
#
#
# # In[ ]:
#
#
# plt.figure(figsize=(6*4,3.5*7))
# for i in range(len(region_names)):
#     plt.subplot(7,4,i+1)
#     plt.plot(1000000*daily_cases[i,:]/population[i], linewidth=2)
#     nvar = np.around(np.var(daily_cases[i,:]/np.amax(daily_cases[i,:])),decimals=3)
#     plt.title(str(region_names[i])+' normalised variance: '+str(nvar))
# plt.show()
#
# plt.figure(figsize=(6*4,3.5*7))
# for i in range(len(region_names)):
#     _temp = np.abs(scipy.fft.fft(daily_cases[i,:]/np.amax(daily_cases[i,:])))
#     _temp = _temp/np.amax(_temp)
#     _temp1 = []
#     n=5
#     for j in range(len(_temp)-n):
#         _temp1.append(np.mean(_temp[j:j+n]))
#     plt.subplot(7,4,i+1)
#     plt.plot(_temp1, linewidth=2)
#     nvar = np.around(np.var(daily_cases[i,:]/np.amax(daily_cases[i,:])),decimals=3)
#     plt.title(str(region_names[i])+' normalised variance: '+str(nvar))
#     plt.xlim([0, 60])
# plt.show()
#
#
# # In[ ]:
#
#
# for i in range(len(region_names)):
#     if region_names[i] == 'BADULLA':
#         idx_kal = i
#     elif region_names[i] == 'COLOMBO':
#         idx_gam = i
#
# print(idx_kal, idx_gam)
#
# list1 = [idx_kal, idx_gam]
#
# # daily_filtered = NO_LPF(daily_cases, datatype='daily',order=1, cutoff=0.1)
# freqs=[]
# for i in range(40):
#       freqs.append((i+1)/200)
#
# print(freqs)
#
#
# # compare gampaha and kalutara
# # overfiltering
# for j in freqs:
#     plt.figure(figsize=(14,3))
#     for i in range(len(list1)):
#         daily_filtered1 = NO_LPF(daily_cases, datatype='daily',order=3, cutoff=j, plot=False)
#         plt.subplot(1,2,i+1)
#         plt.plot(daily_cases[list1[i],:]/np.amax(daily_cases[list1[i],:]), linewidth=2)
#         plt.plot(daily_filtered1[list1[i],:],linewidth=2,color='r')
#         RMSE = np.sqrt(np.mean(np.square(daily_cases[list1[i],:]-daily_filtered1[list1[i],:])))
#         plt.title(str(region_names[list1[i]])+ '  cutoff:  '+str(j))
#         # plt.subplot(1,2,2)
#         # plt.plot(daily_cases[i,:], linewidth=2)
#         # RMSE = np.sqrt(np.mean(np.square(daily_cases[i,:]-daily_filtered[i,:])))
#         # plt.plot(daily_filtered[i,:],linewidth=2,color='r'),plt.title(str(region_names[i])+ ' optimised '+'RMSE= '+str(np.around(RMSE,2)))
#     plt.show()
#
# # # underfiltering
# # for i in list1:
# #   daily_filtered1 = NO_LPF(daily_cases, datatype='daily',order=3, cutoff=0.1, plot=False)
# #   plt.figure(figsize=(6,3.5))
# #   # plt.subplot(1,2,1)
# #   plt.plot(daily_cases[i,:], linewidth=2)
# #   plt.plot(daily_filtered1[i,:],linewidth=2,color='r')
# #   RMSE = np.sqrt(np.mean(np.square(daily_cases[i,:]-daily_filtered1[i,:])))
# #   plt.title(str(region_names[i])+ ' unoptimised '+'RMSE= '+str(np.around(RMSE,2)))
# #   # plt.subplot(1,2,2)
# #   # plt.plot(daily_cases[i,:], linewidth=2)
# #   # RMSE = np.sqrt(np.mean(np.square(daily_cases[i,:]-daily_filtered[i,:])))
# #   # plt.plot(daily_filtered[i,:],linewidth=2,color='r'),plt.title(str(region_names[i])+ ' optimised '+'RMSE= '+str(np.around(RMSE,2)))
# # plt.show()


# # Creating datasets

# ## Preprocessing (Smoothing)

# In[ ]:


def O_LPF(data, datatype, order, R_weight, EIG_weight, corr, region_names, plot_freq=0):
    R_cons = EIG_weight
    EIG_cons = R_weight

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
    step = 0.005
    cutoff_list = range(int(round(1 / step)))
    cutoff_list = 0.1 * (np.array(list(cutoff_list)) + 5) / 100
    # print('cutoff_list=',cutoff_list)

    sections = 7

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
            if corr == True:
                J_R.append(np.mean(np.corrcoef(X, Y)))  # obtaining correlations
            else:
                J_R.append(np.mean(np.square(X - Y)))  # obtaining error

            # obtaining power spectral densities
            X_freqs, X_psd = signal.welch(X)
            Y_freqs, Y_psd = signal.welch(Y)

            X_psd, Y_psd = np.log10(np.abs(X_psd)), np.log10(np.abs(Y_psd))

            # plt.figure()
            # plt.plot(X_psd),plt.plot(Y_psd)

            J0 = []

            # PSD_diff = np.abs(X_psd-Y_psd)
            # inc_fn = np.array(list(range(len(PSD_diff))))**0.5
            # J_eig.append(np.sum(inc_fn*PSD_diff))
            sec_len = int(X_psd.shape[0] / sections)
            for k in range(sections):
                X_avg = np.mean(X_psd[k * sec_len:(k + 1) * sec_len])
                Y_avg = np.mean(Y_psd[k * sec_len:(k + 1) * sec_len])
                J0.append((k + 1) * np.abs(
                    X_avg - Y_avg) ** 0.2)  # eigenvalue spread should increase as k increases for an ideal solution
            J_eig.append(np.sum(J0))

        J_EIG = (J_eig / np.amax(J_eig))

        if corr == True:
            J_E = (J_R / np.amax(J_R)) ** 0.5
        else:
            J_E = 1 - (J_R / np.amax(J_R))

        # J_tot=R_cons*(J_E) +EIG_cons*(J_EIG)
        J_tot = 1 - np.abs(R_cons * (J_E) - EIG_cons * (J_EIG))

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

        # if i % plot_freq == 0:
        #     plt.figure(figsize=(12, 3.5))
        #
        #     plt.subplot(1, 2, 1), plt.title('fitness functions of each component')
        #     plt.plot(cutoff_list, J_E, linewidth=2)
        #     plt.plot(cutoff_list, J_EIG, linewidth=2)
        #     plt.plot(cutoff_list, J_tot, linewidth=2)
        #     plt.xlim([cutoff_list[0], cutoff_list[-1]])
        #     # plt.ylim([0,1.1])
        #     plt.legend(
        #         ['correlation (information retained)', 'eigenvalue spread (noise removed)', 'total fitness function'],
        #         loc='lower left')
        #     plt.xlabel('normalized cutoff frequency')
        #
        #     plt.subplot(1, 2, 2), plt.title(
        #         'cumulative cases in ' + str(region_names[i]) + '\noptimum normalized cutoff frequency: ' + str(
        #             round(cutoff_list[idx], 4)))
        #     plt.plot(X / np.amax(Y), linewidth=2)
        #     plt.plot(Y / np.amax(Y), linewidth=2, color='r')
        #     plt.legend(['original', 'filtered']), plt.xlabel('days')
        #     plt.show()

    return data_filtered.T


# In[ ]:


# confirmed_filtered = O_LPF(confirmed_cases, datatype='daily', order=1, R_cons=1, EIG_cons=1, corr = True)
daily_filtered = O_LPF(daily_cases, datatype='daily', order=3, R_weight=1.0, EIG_weight=1, corr=True,
                       region_names=region_names)


# confirmed_filtered = NO_LPF(confirmed_cases)
# confirmed_filtered = O_NDA(confirmed_cases)
# confirmed_filtered = NO_NDA(confirmed_cases)

# print(confirmed_filtered.shape, confirmed_cases.shape)


# ## Daily cases and per million

# In[ ]:


# @title Run this cell only if we are going from filtered confirmed -> daily filtered.

if not daily_data:

    # fixing the confirmed cases dataset (negative gradients)
    for k in range(confirmed_filtered.shape[0]):
        for i in range(confirmed_filtered.shape[1] - 1):
            if confirmed_filtered[k, i + 1] < confirmed_filtered[k, i]:
                confirmed_filtered[k, i + 1] = confirmed_filtered[k, i]
    confirmed_per_mio_capita = per_million(confirmed_cases, population)
    confirmed_per_mio_capita_filtered = per_million(confirmed_filtered, population)

    daily_cases = get_daily(confirmed_cases)
    daily_filtered = get_daily(confirmed_filtered)

# In[ ]:


daily_per_mio_capita = per_million(daily_cases, population)
daily_per_mio_capita_filtered = per_million(daily_filtered, population)

plots = [daily_cases[:, :].T, daily_filtered[:, :].T]
titles = [DATASET + ': Daily new cases', DATASET + ': Daily new cases(filtered)']
plt.figure(figsize=(12, 3.5))
for i in range(len(titles)):
    plt.subplot(1, 2, i + 1)
    plt.plot(plots[i], linewidth=2)
    plt.title(titles[i]), plt.xlabel('Days since ' + START_DATE)
    # plt.ylim([0,5000])
    plt.ylim([0, 800])

plt.show()

plots = [daily_per_mio_capita[:, :].T, daily_per_mio_capita_filtered[:, :].T]
titles = [DATASET + ': Daily new cases per 1M', DATASET + ': Daily new cases per 1M filtered', ]
plt.figure(figsize=(12, 3.5))
for i in range(len(titles)):
    plt.subplot(1, 2, i + 1)
    plt.plot(plots[i], linewidth=2)
    plt.title(titles[i]), plt.xlabel('Days since ' + START_DATE)
    plt.ylim([0, 800])
plt.show()

if DATASET == 'Texas':
    nums = [50, 125]
    plt.figure(figsize=(12, 3.5))
    for i in range(len(nums)):
        plt.subplot(1, 2, i + 1)
        plt.plot(daily_per_mio_capita[nums[i], :].T, linewidth=1.5)
        plt.plot(daily_per_mio_capita_filtered[nums[i], :].T, linewidth=3)
        # plt.ylim([0,800])
    plt.show

print(daily_per_mio_capita.shape, daily_per_mio_capita_filtered.shape)

# In[ ]:


if DATASET == 'Sri Lanka':

    select_regions = ['COLOMBO', 'GAMPAHA', 'KALUTARA']
    plt.figure(figsize=(len(select_regions) * 6, 4))
    for i in range(len(region_names)):
        for j in range(len(select_regions)):
            if region_names[i] == select_regions[j]:
                plt.subplot(1, len(select_regions), j + 1)
                plt.plot(daily_per_mio_capita[i, :].T, linewidth=2)
                plt.plot(daily_per_mio_capita_filtered[i, :].T, linewidth=3, color='r')
                plt.xlabel('days since 14/11/2020')
                plt.ylabel('daily new cases per million')
                plt.legend([select_regions[j] + ': unfiltered', select_regions[j] + ': filtered'])

# plt.figure(figsize=(18,4))
# for j in range(len(idx)):


# In[ ]:


# ## Creating Alert-Level data

# In[ ]:


places = ['COLOMBO', 'GAMPAHA', 'KALUTARA']
idx = []

for k in range(len(places)):
    for i in range(len(region_names)):
        if region_names[i] == places[k]:
            idx.append(i)

idx


# In[ ]:


def create_alert_level(ts_data, thresholds, logic=1):
    inc_thresh = 7
    dec_thresh = 14
    current = 1
    inc_trig = 0
    dec_trig = 0
    al = []
    for i in range(len(ts_data)):
        if ts_data[i] >= thresholds[current]:
            inc_trig += 1
        else:
            inc_trig = 0
        if ts_data[i] < thresholds[current - 1]:
            dec_trig += 1
        else:
            dec_trig = 0
        if logic == 1:  # high inertia (we dont consider spikes here. we want to see how well it responds. need to quantify it.)
            if inc_trig == inc_thresh:
                current += 1
                inc_trig = 0
            if dec_trig == dec_thresh:
                current -= 1
                dec_trig = 0
        elif logic == 2:
            if np.mean(ts_data[max(0, i - inc_thresh):i]) >= thresholds[current]:
                current += 1
            elif np.mean(ts_data[max(0, i - dec_thresh):i]) < thresholds[current - 1]:
                current -= 1
        elif logic == 3:  # low inertia (we consider spikes here.)
            if ts_data[i] >= thresholds[current]:
                current += 1
            elif ts_data[i] < thresholds[current - 1]:
                current -= 1
        al.append(current)
    return al


alert_filt = []
alert_unfilt = []

# Method 1
thresholds = [0, 10, 20, 40, 1e1000]
daily_f = daily_per_mio_capita_filtered
daily_uf = daily_per_mio_capita
logic = 3

# Method 2
# thresholds = [0,5,10,100,1e1000]
# daily_f = daily_per_mio_capita_filtered/10
# daily_uf = daily_per_mio_capita/10
# logic = 3

for dist in range(daily_f.shape[0]):
    alert_filt.append(create_alert_level(daily_f[dist, :], thresholds, logic=logic))
alert_filt = np.array(alert_filt)

for dist in range(daily_uf.shape[0]):
    alert_unfilt.append(create_alert_level(daily_uf[dist, :], thresholds, logic=logic))
alert_unfilt = np.array(alert_unfilt)


# plt.figure()
# plt.plot(alert_unfilt.T,linewidth=2)
# plt.title("Alert-level for Unfiltered data"),plt.xlabel('Days since '+START_DATE)
# plt.ylim([0, 5])
# plt.show()

# plt.figure()
# plt.plot(alert_filt.T,linewidth=2)
# plt.title("Alert-level for Filtered data"),plt.xlabel('Days since '+START_DATE)
# plt.ylim([0, 5])
# plt.show()

# count \/ from alert levels
def count_spikes(data):
    count_total = np.zeros([data.shape[0], ])
    for d in range(data.shape[0]):
        i = 0
        count = 0
        while i < (data.shape[1] - 2):
            if data[d, i] - data[d, i + 1] == 1 and data[d, i + 1] - data[d, i + 2] == -1:
                count = count + 1
            elif data[d, i] - data[d, i + 1] == -1 and data[d, i + 1] - data[d, i + 2] == 1:
                count = count + 1
            i = i + 1
        count_total[d] = count
    return count_total


count_unfilt = np.sum(count_spikes(alert_unfilt))
print('count_unfiltered=', count_unfilt)

count_filt = np.sum(count_spikes(alert_filt))
print('count_filtered=', count_filt)

# # removing \/ from alert levels
# def remove_spikes(data):
#     for d in range(data.shape[0]):
#         for i in range(data.shape[1]-2):
#             if data[d,i] - data[d,i+1] == 1 and data[d,i+1] - data[d,i+2] == -1:
#                 data[d,i+1] = data[d,i]
#     return data

# alert_unfilt = remove_spikes(alert_unfilt)
# alert_filt = remove_spikes(alert_filt)

# plt.figure()
# plt.plot(alert_unfilt.T,linewidth=2)
# plt.title("Alert-level for Unfiltered data after filtering spikes"),plt.xlabel('Days since '+START_DATE)
# plt.ylim([0, 5])
# plt.show()

# plt.figure()
# plt.plot(alert_filt.T,linewidth=2)
# plt.title("Alert-level for Filtered data after filtering spikes"),plt.xlabel('Days since '+START_DATE)
# plt.ylim([0, 5])
# plt.show()

alert_f = np.copy(alert_filt)
alert_uf = np.copy(alert_unfilt)

for i in range(len(thresholds) - 1):
    alert_f[alert_f == i + 1] = thresholds[i]
    alert_uf[alert_uf == i + 1] = thresholds[i]

# plt.figure(figsize=(5*5,3*5))
# for i in range(alert_filt.shape[0]):
#   plt.subplot(5,5,i+1)
#   plt.plot(alert_uf[i,:],linewidth=2,color='b')
#   plt.plot(alert_f[i,:],linewidth=2,color='r')
#   plt.plot(daily_uf[i,:],linewidth=1.5,color='y')
#   plt.plot(daily_f[i,:],linewidth=1.5,color='g')
#   plt.xlim([30,120])
# plt.show()
i = 13

plt.plot(daily_uf[i, :], 'b:')
plt.plot(daily_f[i, :], 'r:')
plt.plot(alert_uf[i, :], linewidth=2, color='b')
plt.plot(alert_f[i, :], linewidth=2, color='r')
plt.legend(["original epicurve", "smoothed epicruve", "levels computed using original data (scaled)",
            "levels computed using smoothed data (scaled)"])
plt.xlabel('days since 14 Nov 2020')
plt.ylabel('daily cases')
plt.xlim([30, 120])
plt.ylim([0, thresholds[-2] + 20])
plt.show()

print(i)
print(region_names[i])

# In[ ]:


alert_f = np.copy(alert_filt)
alert_uf = np.copy(alert_unfilt)

idx = np.random.randint(0, 24, size=(3,))

plt.figure()
for i in range(len(idx)):
    if i == 0:
        plt.plot(alert_uf[idx[i], :], 'b:')
        plt.plot(alert_f[idx[i], :], linewidth=2, color='b')
    elif i == 1:
        plt.plot(alert_uf[idx[i], :], 'r:')
        plt.plot(alert_f[idx[i], :], linewidth=2, color='r')
    else:
        plt.plot(alert_uf[idx[i], :], 'g:')
        plt.plot(alert_f[idx[i], :], linewidth=2, color='g')

plt.legend(
    [region_names[idx[0]] + ': computed using original data', region_names[idx[0]] + ': computed using filtered data',
     region_names[idx[1]] + ': computed using original data', region_names[idx[1]] + ': computed using filtered data',
     region_names[idx[2]] + ': computed using original data', region_names[idx[2]] + ': computed using filtered data'])

plt.show()
print(idx)

# # GSP

# ## coordinate stuff

# In[ ]:
#
#
# region_codes = features.index
# dpmc = pd.Series(daily_cases.mean(axis=1), name="Mean Daily Per Million")
# names = pd.Series(region_codes, name="District")
# state_data = pd.concat([names, dpmc], 1)
# print(state_data)
#
# # In[ ]:
#
#
# state_geo = "maps/LKA_electrol_districts.geojson"
#
# m = folium.Map(location=[8, 81], zoom_start=8)
# folium.GeoJson(state_geo, name="geojson").add_to(m)
#
# folium.Choropleth(
#     geo_data=state_geo,
#     name="choropleth",
#     data=state_data,
#     columns=["District", "Mean Daily Per Million"],
#     key_on="properties.electoralDistrictCode",
#     fill_color="YlGn",
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     legend_name="Mean Covid-19 Cases",
# ).add_to(m)
#
# folium.LayerControl().add_to(m)
#
# # Averaging corrdinates of the map to find the longitudes and latitudes of each region
#
# # In[ ]:
#
#
# import json
#
# map_json = json.load(open(state_geo))
#
# dist_coor = {"Code": [], "lon": [], "lat": []}
# for i in range(len(map_json['features'])):
#     coor = np.array(map_json['features'][i]['geometry']['coordinates'])
#     if len(coor.shape) == 3:
#         mean_coor = np.mean(coor, 1)[0]
#     else:
#         mean_coor = []
#         for j in range(coor.shape[0]):
#             # print(np.shape(coor[j][0]))
#             mean_coor.append(np.mean(coor[j][0], 0))
#         mean_coor = np.mean(mean_coor, 0)
#     dist_coor["Code"] += [map_json['features'][i]['properties']['electoralDistrictCode']]
#     dist_coor["lon"] += [mean_coor[0]]
#     dist_coor["lat"] += [mean_coor[1]]
# dist_coor = pd.DataFrame(dist_coor).set_index('Code').sort_index()
#
# print(dist_coor.index)
# print(region_codes)
#
#
# # ## GSP_old
#
# # In[ ]:
#
#
# def laplacian(A, method=0, plot=False):
#     """
#     A : adjacency matrix
#     method : 0 - combinatorial, 1 - normalized
#     """
#     A_hat = A + np.identity(A.shape[0])
#     D_hat = np.zeros_like(A)
#     for i in range(D_hat.shape[0]):
#         D_hat[i, i] = np.sum(A_hat[i, :])
#
#     if method == 0:
#         L = D_hat - A_hat
#     elif method == 1:
#         D1 = np.sqrt(np.linalg.inv(D_hat))
#         A1 = np.matmul(D1, A_hat)
#         L = np.identity(A_hat.shape[0]) - np.matmul(A1, D1)
#     elif method == 2:
#         L = spektral.utils.gcn_filter(A, symmetric=True)
#     if plot == True:
#         plots, names = np.array([A_hat, D_hat, L]), ['adjacency', 'degree matrix', 'laplacian']
#         plt.figure(2, figsize=(16, 4))
#         for i in range(plots.shape[0]):
#             plt.subplot(1, 4, i + 1), plt.title(names[i]), plt.imshow(plots[i, 0:50, 0:50]), plt.colorbar()
#     return L
#
#
# # In[ ]:
#
#
# dist_coor = features[['Lat', "Lon"]]
# dist_coor
#
# A = convert_lon_lat_to_adjacency_matrix(dist_coor, lat_column="Lat", lon_column="Lon", delta=1e-5)
# plt.imshow(A)
# plt.show()
#
# # In[ ]:
#
#
# L = laplacian(A, 0, True)
#
#
# # ## GSP_new
#
# # In[ ]:
#
#
# def adjacency_mat(X, type_, cutoff, self_loops, is_weighted, plot):
#     nodes = X.shape[0]
#     d1 = np.zeros([nodes, nodes])
#     d = np.zeros(X.shape[1])
#     for i in range(nodes):
#         for j in range(nodes):
#             for k in range(X.shape[1]):
#                 d[k] = X[i, k] - X[j, k]
#             d1[i, j] = np.sqrt(np.sum(np.square(d)))
#     dist = d1 / np.amax(d1)
#     if type_.lower() in ['gaussean']:
#         theta = np.mean(dist)
#         A_dense = -1 * np.square(dist) / (0.3 * np.square(theta))
#         A_dense = np.exp(A_dense)
#     else:
#         A_dense = 1 * dist
#
#     if is_weighted == False:
#         A_dense[A_dense != 0] = 1
#     if self_loops == False:
#         A_dense = A_dense - np.identity(nodes)
#     A = 1 * A_dense
#     A[dist > cutoff] = 0
#     if plot == True:
#         plots = np.array([dist, A_dense, A])
#         titles = ['distance matrix', 'adjacency matrix: dense (without cutoff)',
#                   'adjacency matrix: sparse (with cutoff)']
#         plt.figure(figsize=(12, 4))
#         for i in range(len(plots)):
#             plt.subplot(1, 3, i + 1), plt.imshow(plots[i, 0:25, 0:25]), plt.title(titles[i])
#         plt.show()
#     return A
#
#
# def laplacian(A, type_, plot):
#     D = np.zeros_like(A)
#     D_hat = np.zeros_like(A)
#     A_hat = 1 * A
#     if A[0, 0] == 1:
#         for i in range(A.shape[0]):
#             A[i, i] = 0
#     else:
#         for i in range(A.shape[0]):
#             A_hat[i, i] = 1
#     for i in range(D.shape[0]):
#         D[i, i] = np.sum(A[i, :])
#         D_hat[i, i] = np.sum(A_hat[i, :])
#     if type_ == 'combinatorial':
#         L = D - A
#     elif type_ == 'normalized':
#         D = np.sqrt(np.linalg.inv(D))
#         L = np.identity(A.shape[0]) - np.matmul(np.matmul(D, A), D)
#     if plot == True:
#         plots, names = np.array([A_hat, D, L]), ['Adjacency matrix with self loops', 'Degree', 'Laplacian']
#         plt.figure(figsize=(12, 4))
#         for i in range(plots.shape[0]):
#             plt.subplot(1, 3, i + 1), plt.title(names[i]), plt.imshow(plots[i, 0:25, 0:25])
#         plt.show()
#     return L
#
#
# # In[ ]:
#
#
# # GETTING THE COORDINATES for SL
# dist_coor = features[['Lat', "Lon"]]
# coordinates = np.float64(np.array(dist_coor.iloc[:, :]))
#
# # obtain weight matrices to convert into adjacency matrix
# cutoff = 0.2
# dist_type = 'gaussean'
# laplacian_type = 'normalized'
# A = adjacency_mat(coordinates, dist_type, cutoff, self_loops=False, is_weighted=True, plot=True)
# L = laplacian(A, laplacian_type, plot=True)
#
# # In[ ]:
#
#
# """
# eigenvectors of laplacian
# """
# [eig_L, v_L] = np.linalg.eig(L)
#
# plt.plot(eig_L)
#
# eig_L.shape, v_L.shape
#
# # In[ ]:
#
#
# # state_var = confirmed_per_mio_capita_filtered
# state_var = daily_per_mio_capita_filtered
# state_var.shape
#
# """
# total variation of graph signal (smoothness measure)
# """
#
# total_variation = np.zeros((state_var.shape[1],))
#
# for i in range(state_var.shape[1]):
#     temp = np.matmul(L, state_var[:, i])
#     total_variation[i] = np.matmul(state_var[:, i].T, temp)
#
# """
# total variation of eigenvectors (signal energy)
# """
# total_variation_eig = np.zeros((v_L.shape[0],))
#
# for i in range(v_L.shape[0]):
#     temp = np.matmul(L, v_L[:, i])
#     total_variation_eig[i] = np.matmul(v_L[:, i].T, temp)
#
# """
# graph fourier transform
# """
# state_var_GFT = np.zeros_like(state_var)
#
# for i in range(state_var_GFT.shape[1]):
#     state_var_GFT[:, i] = np.matmul(v_L.T, state_var[:, i])
#
# """
# lowpass filters
# """
# filter_size = 10
# filter_start = 0
#
# H_L = np.zeros((filter_size))
#
# # In[ ]:
#
#
# plt.plot(total_variation)

# # TRAINING MODEL

# In[ ]:


# TRAINING PARAMETERS

# TRAINING MODEL
batch_size = 16
epochs = 100
lr = 0.002
seq_size = 14
predict_steps = 6
n_dem = 16
s_per_example = 30
n_features = 4

# ## Dense Models

# In[ ]:


reduce_regions2batch = True
model_type = "DENSE"


def get_model_ANN(seq_size, predict_steps, n_features=n_features, n_regions=n_regions):
    inp_seq = tf.keras.layers.Input(seq_size, name="input_sequence")
    inp_fea = tf.keras.layers.Input(n_features, name="input_features")

    x = inp_seq
    xf = inp_fea
    n = n_features
    while (n > 0):
        xf = tf.keras.layers.Dense(n, activation='relu')(xf)
        n = n // 2

    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(predict_steps, activation='relu')(x)

    if n_features > 0:
        x = x * xf
    model = tf.keras.models.Model([inp_seq, inp_fea], x)
    return model


model_ANN = get_model_ANN(seq_size, predict_steps)
model_ANN.summary()
tf.keras.utils.plot_model(model_ANN, show_shapes=True, rankdir='LR')

# In[ ]:


# In[ ]:


reduce_regions2batch = False
model_type = "DENSE"


def get_model_ANN(input_seq_size, output_seq_size, n_features=n_features, n_regions=n_regions):
    inp_seq = tf.keras.layers.Input((input_seq_size, n_regions), name="input_sequence")
    inp_fea = tf.keras.layers.Input((n_features, n_regions), name="input_features")

    x = tf.keras.layers.Reshape((input_seq_size * n_regions,))(inp_seq)
    x = tf.keras.layers.Dense(output_seq_size * n_regions, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((output_seq_size, n_regions))(x)

    model = tf.keras.models.Model([inp_seq, inp_fea], x)
    return model


model_ANN = get_model_ANN(seq_size, predict_steps, n_regions=n_regions)
model_ANN.summary()
tf.keras.utils.plot_model(model_ANN, show_shapes=True, rankdir='LR')

# ## LSTM Models

# In[ ]:


reduce_regions2batch = True
model_type = "LSTM"


def get_model_LSTM(input_seq_size, output_seq_size, n_features=n_features, n_regions=n_regions):
    inp_seq = tf.keras.layers.Input((input_seq_size, 1), name="input_seq")
    inp_fea = tf.keras.layers.Input(n_features, name="input_fea")

    out = tf.keras.layers.LSTM(output_seq_size, activation='sigmoid')(inp_seq)

    model = tf.keras.models.Model([inp_seq, inp_fea], out)
    return model


model_LSTM = get_model_LSTM(seq_size, predict_steps)
model_LSTM.summary()
tf.keras.utils.plot_model(model_LSTM, show_shapes=True, rankdir='LR')

# In[ ]:


reduce_regions2batch = False
model_type = "LSTM"


def get_model_LSTM(input_seq_size, output_seq_size, n_regions):
    inp_seq = tf.keras.layers.Input((input_seq_size, n_regions), name="input_seq")
    x = tf.keras.layers.LSTM(output_seq_size * n_regions, activation='sigmoid')(inp_seq)
    x = tf.keras.layers.Reshape((output_seq_size, n_regions))(x)

    model = tf.keras.models.Model(inp_seq, x)

    return model


model_LSTM = get_model_LSTM(seq_size, predict_steps, n_regions)
model_LSTM.summary()
tf.keras.utils.plot_model(model_LSTM, show_shapes=True, rankdir='LR')

# In[ ]:


reduce_regions2batch = False
model_type = "LSTM_MULTI"


def get_model_LSTM(input_seq_size, output_seq_size, n_regions):
    inp_seq = tf.keras.layers.Input((input_seq_size, n_regions), name="input_seq")

    lstm_input = inp_seq
    for i in range(output_seq_size):
        xx = tf.keras.layers.LSTM(n_regions, activation='relu')(lstm_input)
        if i == 0:
            out = xx
        else:
            out = tf.keras.layers.concatenate([out, xx])

        xx = tf.reshape(xx, (-1, 1, n_regions))
        lstm_input = tf.keras.layers.concatenate([lstm_input[:, 1:, :], xx], axis=1)

    out = tf.reshape(out, (-1, output_seq_size, n_regions))
    model = tf.keras.models.Model(inp_seq, out)
    return model


model_LSTM_multi = get_model_LSTM(seq_size, predict_steps, n_regions)
model_LSTM_multi.summary()
tf.keras.utils.plot_model(model_LSTM_multi, show_shapes=True, rankdir='TB')


# ## Select Training model

# In[ ]:


def reset():
    if model_type == "DENSE":
        model = get_model_ANN(seq_size, predict_steps, n_regions)
    elif model_type == "LSTM":
        model = get_model_LSTM(seq_size, predict_steps, n_regions)
    elif model_type == "LSTM_MULTI":
        model = get_model_LSTM(seq_size, predict_steps, n_regions)
    loss_f = tf.keras.losses.MeanSquaredError()
    opt = Adam(lr=lr)
    return model, loss_f, opt


def eval_metric(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) ** 0.5


model, loss_f, opt = reset()
model.summary()


# ## Select training data
# 
# Run the required cell

# ### FILTERED DATA

# In[ ]:


def get_data_daily():
    x_data = np.copy(daily_per_mio_capita_filtered)
    y_data = np.copy(daily_per_mio_capita_filtered)
    # y_data = np.copy(alert_unfilt)
    return x_data, y_data


def get_data_confirmed():
    x_data = np.copy(confirmed_per_mio_capita_filtered)
    y_data = np.copy(confirmed_per_mio_capita_filtered)
    # y_data = np.copy(alert_unfilt)
    return x_data, y_data


TRAINING_DATA_TYPE = "Filtered"


# ### UNFILTERED DATA

# In[ ]:


def get_data_daily():
    x_data = np.copy(daily_per_mio_capita)
    y_data = np.copy(daily_per_mio_capita)
    # y_data = np.copy(alert_unfilt)
    return x_data, y_data


def get_data_confirmed():
    x_data = np.copy(confirmed_per_mio_capita)
    y_data = np.copy(confirmed_per_mio_capita)
    # y_data = np.copy(alert_unfilt)
    return x_data, y_data


TRAINING_DATA_TYPE = "Unfiltered"


# ## Spliting and reshaping data

# ### Normalizing functions

# In[ ]:


# ==================================== TODO: find a good normalization technique
def normalize_for_nn(data, given_scalers=None):
    print(f"NORMALIZING; Data: {data.shape}")
    data = data.astype('float32')
    scalers = []
    for i in range(data.shape[0]):
        if given_scalers is not None:
            scale = given_scalers[i]
        else:
            scale = float(np.max(data[:, :]))
        scalers.append(scale)
        data[i, :] /= scale
    return data, scalers


def undo_normalization(normalized_data, scalers):
    print(f"UNNORMALIZING; Norm Data: {normalized_data.shape}")
    for i in range(len(scalers)):
        normalized_data[:, :, i] *= scalers[i]
    return normalized_data


# ### Split dataset on region dimension.

# In[ ]:


if daily_data == True:
    x_data, y_data = get_data_daily()
else:
    x_data, y_data = get_data_confirmed()

x_data, x_data_scalers = normalize_for_nn(x_data)
y_data, x_data_scalers = normalize_for_nn(y_data, x_data_scalers)

plt.figure()
plt.subplot(121)
plt.plot(x_data.T), plt.title("X data")
plt.subplot(122)
plt.plot(y_data.T), plt.title("Y data")
plt.show()

plt.figure()
plt.subplot(121)
plt.hist(x_data.reshape(-1)), plt.title("X data")
plt.subplot(122)
plt.hist(y_data.reshape(-1)), plt.title("Y data")
plt.show()

# shuffle regions so that we don't get the same k fold everytime
index = np.arange(x_data.shape[0])
np.random.shuffle(index)
x_data = x_data[index]
y_data = y_data[index]
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_on_region_dimension(x_data, y_data, seq_size, predict_steps,
                                                                           s_per_example,
                                                                           k_fold=5, test_fold=1,
                                                                           reduce_last_dim=reduce_regions2batch)

print("Train", X_train.shape, Y_train.shape)
print("Val", X_val.shape, Y_val.shape)
print("Test", X_test.shape, Y_test.shape)

# ### Split dataset on time dimension.

#  First, we split the dataset into a subset called the training set, and another subset called the test set. If any parameters need to be tuned, we split the training set into a training subset and a validation set. The model is trained on the training subset and the parameters that minimize error on the validation set are chosen. Finally, the model is trained on the full training set using the chosen parameters, and the error on the test set is recorded.

# ![kf.png](data:image/png;base64,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)

# When dealing with time series data, traditional cross-validation (like k-fold) should not be used for two reasons:Temporal Dependencies,Arbitrary Choice of Test Set

# ![kf.png](data:image/png;base64,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)

# In[ ]:


if daily_data:
    x_data, y_data = get_data_daily()
else:
    x_data, y_data = get_data_confirmed()

x_data, x_data_scalers = normalize_for_nn(x_data)
y_data, x_data_scalers = normalize_for_nn(y_data, x_data_scalers)

plt.figure()
plt.subplot(121)
plt.plot(x_data.T), plt.title("X data")
plt.subplot(122)
plt.plot(y_data.T), plt.title("Y data")
plt.show()

plt.figure()
plt.subplot(121)
plt.hist(x_data.reshape(-1)), plt.title("X data")
plt.subplot(122)
plt.hist(y_data.reshape(-1)), plt.title("Y data")
plt.show()

# shuffle regions so that we don't get the same k fold everytime (Should we do this???)
# index = np.arange(x_data.shape[0])
# np.random.shuffle(index)
# x_data = x_data[index]
# y_data = y_data[index]
X_train, X_train_feat, Y_train, X_val, X_val_feat, Y_val, X_test, X_test_feat, Y_test = split_on_time_dimension(x_data,
                                                                                                                y_data,
                                                                                                                features,
                                                                                                                seq_size,
                                                                                                                predict_steps,
                                                                                                                k_fold=3,
                                                                                                                test_fold=2,
                                                                                                                reduce_last_dim=reduce_regions2batch)

print("Train", X_train.shape, Y_train.shape, X_train_feat.shape)
print("Val", X_val.shape, Y_val.shape, X_val_feat.shape)
print("Test", X_test.shape, Y_test.shape, X_test_feat.shape)

# In[ ]:


x_data.shape, y_data.shape, X_train.shape, Y_train.shape

# ## undersampling
# currently it works when dataset is split on region dimension. i.e data.shape = [regions, samples, sample size]
# 
# use count_power = *value* to set how it should undersample to bias either sample count or dataset balance

# In[ ]:


import random


def get_count(segments, data):
    bounds = []
    count = []
    idx = []
    for i in range(segments):
        data = (data - np.amin(data))
        bounds.append(np.round((i + 1) * np.amax(data) / segments, 3))
        if i == 0:
            ineq = data <= bounds[i]
        elif i == (segments - 1):
            ineq = data > bounds[i - 1]
        else:
            ineq = (data > bounds[i - 1]) * (data <= bounds[i])
        count.append(np.sum(ineq))
        idx.append(np.reshape(np.array(np.where(ineq)), [-1, ]))
    count = np.array(count).astype(int)
    bounds = np.array(bounds).astype(np.float64)
    return count, bounds, idx


# In[ ]:


samples_all = np.transpose(X_train, [2, 0, 1])
X_train_reshaped = np.transpose(X_train, [2, 0, 1])
Y_train_reshaped = np.transpose(Y_train, [2, 0, 1])

count_power = 1
plot_state = 1

samples_mean = np.zeros([samples_all.shape[0], samples_all.shape[1]])
# evaluating optimal number of segments for each district
segment_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]
segment_dist = []
if plot_state == 1:
    plt.figure(figsize=(5 * 6, 5 * 4))
for i in range(samples_all.shape[0]):
    for k in range(samples_all.shape[1]):
        samples_mean[i, k] = np.mean(samples_all[i, k, :])
    all_counts = []
    count_score = []
    # evaluating the count score for each district
    for n in range(len(segment_array)):
        segments = segment_array[n]
        [count, bounds, idx] = get_count(segments, samples_mean[i, :])
        all_counts.append(np.amin(count) * len(count))
        count_score.append((all_counts[n] ** count_power) * (n + 1))
    if plot_state == 1:
        plt.subplot(5, 5, i + 1)
        plt.plot(segment_array, all_counts / np.amax(all_counts), linewidth=2)
        plt.plot(segment_array, count_score / np.amax(count_score), linewidth=2)
        plt.legend(['normalised total counts', 'segment score'])
        plt.title('dist: ' + region_names[i] + '  segments: ' + str(
            segment_array[np.argmax(count_score)]) + '  samples: ' + str(all_counts[np.argmax(count_score)]))
    segment_dist.append(segment_array[np.argmax(count_score)])
segment_dist = np.array(segment_dist).astype(int)
if plot_state == 1:
    plt.show()
print('segments per district= ', segment_dist)

# #### output shape:  data[region][sample][length].
# Now we have a list (not numpy) with undersampled train and test data. We may have to convert it to a numpy array at the training phase itself.

# In[ ]:


# GETTING RANDOM INDICES AND UNDERSAMPLING WITH OPTIMAL NUMBER OF SAMPLES
idx_rand_all = []
X_undersampled = []
Y_undersampled = []
for i in range(samples_all.shape[0]):
    data = samples_mean[i, :]
    segments = segment_dist[i]
    [count_dist, bounds_dist, idx_dist] = get_count(segments, data)
    n_per_seg = np.amin(count_dist)
    data_new = []
    idx_rand = np.zeros([segments, n_per_seg])
    for k in range(segments):
        idx_temp = list(idx_dist[k])
        idx_rand[k, :] = random.sample(idx_temp, n_per_seg)
    idx_rand = np.reshape(idx_rand, [-1, ]).astype(int)

    X_undersampled.append(X_train_reshaped[i, idx_rand, :])
    Y_undersampled.append(Y_train_reshaped[i, idx_rand, :])
    idx_rand_all.append(idx_rand)

# In[ ]:


region_ = 'COLOMBO'
for i in range(len(region_names)):
    if region_names[i] == region_:
        idx = i

plt.figure(figsize=(6 * 4, 6 * 2))
for i in range(len(X_undersampled[idx])):
    plt.subplot(6, 6, i + 1)
    plt.plot(X_undersampled[idx][i])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.title('index: ' + str(idx_rand_all[idx][i]))
plt.suptitle('all samples for ' + region_names[idx])
plt.show()

# ## Training phase

# In[ ]:


k_fold = 5

# In[ ]:


folder = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime()) + "_" + DATASET
os.makedirs('./logs/' + folder)
tensorboard = TensorBoard(log_dir='./logs/' + folder, write_graph=True, histogram_freq=1, write_images=True)
tensorboard.set_model(model)

train_metric = []
val_metric = []
test_metric = []

if daily_data == True:
    x_data, y_data = get_data_daily()
else:
    x_data, y_data = get_data_confirmed()

x_data, x_data_scalers = normalize_for_nn(x_data)
y_data, x_data_scalers = normalize_for_nn(y_data, x_data_scalers)

# shuffle regions so that we don't get the same regions order everytime  when training
index = np.arange(x_data.shape[0])
np.random.shuffle(index)
x_data = x_data[index]
y_data = y_data[index]
best_test_value = 1e10
for test_fold in range(k_fold - 1, k_fold):
    print(f"************* k fold: {test_fold}/{k_fold}")
    model_name = DATASET + "_" + model_type + "_" + TRAINING_DATA_TYPE + "_fold_" + str(test_fold)
    # model, loss_f, opt = reset()

    # X_train, Y_train, X_val, Y_val, X_test, Y_test = split_on_time_dimension(x_data, y_data, seq_size, predict_steps, s_per_example, 
    #                                                      k_fold=k_fold, test_fold=test_fold, reduce_last_dim=reduce_regions2batch)

    # for epoch in range(epochs):
    #     losses = []
    #     for i in range(0, len(X_train), batch_size):
    #         if i + batch_size > len(X_train):
    #             continue
    #         x = X_train[i:i+batch_size]
    #         y = Y_train[i:i+batch_size]
    #         # compute loss for that batch
    #         with tf.GradientTape() as tape:
    #             y_pred = model(x, training=True)
    #             loss = tf.reduce_mean(loss_f(y, y_pred))
    #         # compute d(loss)/d(weight)
    #         grad = tape.gradient(loss, model.trainable_variables)
    #         # gradients applied to optimizer
    #         opt.apply_gradients(zip(grad, model.trainable_variables))

    #         losses.append(loss)

    #         # val loss
    #         val_loss = loss_f(Y_val, model(X_val, training=False))

    #         print(f"\r Epoch {epoch}: mean loss = {np.mean(losses):.5f} mean val loss = {np.mean(val_loss):.5f}", end='')

    #     # add metric value of the prediction (from training data)
    #     pred_train_y = model(X_train, training=False)
    #     train_metric.append(eval_metric(Y_train,pred_train_y))

    #     pred_val_y = model(X_val, training=False)
    #     val_metric.append(eval_metric(Y_val,pred_val_y))

    #     # add metric value of the prediction (from testing data)
    #     pred_test_y = model(X_test, training=False)
    #     test_metric.append(eval_metric(Y_test,pred_test_y))

    #     tensorboard.on_epoch_end(epoch, {"loss": np.mean(losses)})

    #     #   plt.subplot(122)
    #     #   if len(model.output.shape) == 2:
    #     #       Y_pred = np.zeros_like(Y_test)
    #     #       Y_pred[:,:,i] = model(X_test[:,:,i])
    #     #   else:
    #     #       Y_pred = model(X_test).numpy()
    #     #   plot_prediction(X_test, Y_test, Y_pred, region_names)
    #     #   plt.setp(plt.gca().get_legend().get_texts(), fontsize='5') # for legend text
    #     #   plt.setp(plt.gca().get_legend().get_title(), fontsize='5') # for legend title
    #     #   plt.show()
    # plt.figure(figsize=(10,3))
    # plt.subplot(121)
    # plt.ion()
    # plt.plot(train_metric, label='Train')
    # plt.plot(val_metric, label='Validation')
    # plt.plot(test_metric, label='Test')
    # plt.xlabel("Epoch")
    # plt.ylabel("Metric")
    # plt.legend()
    # plt.ioff()
    # plt.show()
    # train_metric = []
    # val_metric = []
    # test_metric = []

    # 
    # model.save(model_name+".h5")

    print("Adjusting hyperparameters ... ")

    print("Training using training+validation data.")
    model, loss_f, opt = reset()

    X_train, Y_train, X_train_feat = np.concatenate([X_train, X_val], 0), np.concatenate([Y_train, Y_val],
                                                                                         0), np.concatenate(
        [X_train_feat, X_val_feat], 0)
    for epoch in range(epochs):
        losses = []
        for i in range(0, len(X_train), batch_size):
            if i + batch_size > len(X_train):
                continue
            x = X_train[i:i + batch_size]
            y = Y_train[i:i + batch_size]

            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = tf.reduce_mean(loss_f(y, y_pred))

            grad = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grad, model.trainable_variables))
            losses.append(loss)

            print(f"\r Epoch {epoch}: mean loss = {np.mean(losses):.5f}", end='')

        # add metric value of the prediction (from training data)
        pred_train_y = model(X_train, training=False)
        train_metric.append(eval_metric(Y_train, pred_train_y))
        # add metric value of the prediction (from testing data)
        pred_test_y = model(X_test, training=False)
        test_metric.append(eval_metric(Y_test, pred_test_y))
        if test_metric[-1] < best_test_value:
            best_test_value = test_metric[-1]
            model.save("temp.h5")
            print(f"Best test metric {best_test_value}. Saving model...")

    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    plt.ion()
    plt.plot(train_metric, label='Train')
    plt.plot(test_metric, label='Test')
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.ioff()
    plt.show()
    train_metric = []
    test_metric = []
model = tf.keras.models.load_model("temp.h5")
fmodel_name = DATASET + "_" + model_type + "_" + TRAINING_DATA_TYPE
model.save("models/" + fmodel_name + ".h5")

# ## Loading a model from drive

# In[ ]:


print(os.getcwd())
model_name = input("Give path to the model: ")
model = tf.keras.models.load_model(model_name)

model_name_ = model_name.split("_")
DATASET = model_name_[0]
model_type = model_name_[1]
TRAINING_DATA_TYPE = model_name_[2]


# # Evaluating models

# Our strategy to get the results
# 
# ---
# 
# 
# 
# 1.   Train filtered data - Test filtered data
#     1. Compare predicted filtered with raw filtered data
#     2. Compare predicted filtered with raw unfiltered data
# 2.   Train unfiltered data - Test unfiltered data
#     1. Compare predicted unfiltered with **raw filtered data*** (we are saying that filtered data is smooth/generally good. the idea of predicting unfiltered data is that it is much closer to ground truth of unfiltered data. Therefore no point of comparing with filtered data.)
#     2. Compare predicted unfiltered with raw unfiltered data
# 
# *we use unfiltered data for eveything except for evaluation 
# 

# ## Helper functions for evaluation

# In[ ]:


def eval_model(model, x_data_raw, y_data_raw, dis=18, predict_data_label="Unfiltered"):
    x_data, x_data_scalers = normalize_for_nn(x_data_raw)
    y_data, x_data_scalers = normalize_for_nn(y_data_raw, x_data_scalers)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_on_time_dimension(x_data, y_data, seq_size, predict_steps,
                                                                             s_per_example,
                                                                             k_fold=k_fold, test_fold=k_fold - 1,
                                                                             reduce_last_dim=False)
    if len(model.output.shape) == 2:
        Y_pred = np.zeros_like(Y_test)
        for i in range(len(region_names)):
            Y_pred[:, :, i] = model(X_test[:, :, i])
    else:
        Y_pred = model(X_test).numpy()

    Y_test = undo_normalization(Y_test, x_data_scalers)
    Y_pred = undo_normalization(Y_pred, x_data_scalers)

    rmse = np.mean((Y_test - Y_pred) ** 2) ** 0.5
    mae = np.mean((np.abs(Y_test - Y_pred)))
    print(f"RMSE={rmse:.2f} MAE={mae:.2f}")

    # CREATING TRAIN-TEST SETS FOR CASES
    X_test, Y_test = split_into_pieces_inorder(x_data, y_data, seq_size, predict_steps, s_per_example,
                                               seq_size + predict_steps, reduce_last_dim=False)

    if len(model.output.shape) == 2:
        Y_pred = np.zeros_like(Y_test)
        for i in range(len(region_names)):
            Y_pred[:, :, i] = model(X_test[:, :, i])
    else:
        Y_pred = model(X_test).numpy()

    # NOTE:
    # max value may change with time. then we have to retrain the model!!!!!!
    # we can have a predefined max value. 1 for major cities and 1 for smaller districts
    X_test = undo_normalization(X_test, x_data_scalers)
    Y_test = undo_normalization(Y_test, x_data_scalers)
    Y_pred = undo_normalization(Y_pred, x_data_scalers)

    print(x_data_scalers)
    tmp = []
    tmpgt = []

    for i in range(len(X_test)):
        for j in range(seq_size):
            tmp.append(X_test[i, j, dis])
            tmpgt.append(X_test[i, j, dis])
        for j in range(predict_steps):
            tmp.append(Y_pred[i, j, dis])
            tmpgt.append(Y_test[i, j, dis])

    plt.figure(figsize=(10, 10))
    plt.plot(tmpgt)
    plt.plot(tmp)
    plt.plot(y_data_raw[dis, :], '--')
    plt.plot(x_data_raw[dis, :], '--')
    plt.legend(['ground truth', 'forecast', 'all GT data', 'all Input data'])
    plt.title(f"Sample district {region_names[dis]}"), plt.xlabel("Days"), plt.ylabel("Cases")
    plt.show()

    region_mask = (np.mean(x_data_raw, 1) > 80).astype('int32')

    plt.figure(figsize=(20, 10))
    plot_prediction2(X_test, Y_test, Y_pred, region_names, region_mask)
    plt.savefig(f"images/{model_type}_{TRAINING_DATA_TYPE}_{predict_data_label}.eps")
    plt.savefig(f"images/{model_type}_{TRAINING_DATA_TYPE}_{predict_data_label}.jpg")
    plt.show()


# ## Evaluate on unfiltered data

# In[ ]:


print(f"=================================== Trained on {TRAINING_DATA_TYPE} data. Evaluating on Unfiltered data")

if daily_data == True:
    x_data, y_data = get_data_daily()
    y_data = np.copy(daily_per_mio_capita)
else:
    x_data, y_data = get_data_confirmed()
    y_data = np.copy(confirmed_per_mio_capita)
# y_data = np.copy(alert_unfilt) 

plt.plot(x_data.T), plt.title("Input original data"), plt.xlabel("Days"), plt.ylabel("Cases"), plt.show()
plt.plot(y_data.T), plt.title("Expecting prediction (GT)"), plt.xlabel("Days"), plt.ylabel("Cases"), plt.show()

eval_model(model, x_data, y_data, dis=0, predict_data_label="Unfiltered")

# ## Evaluate on Filtered data

# In[ ]:


print(f"================================= Trained on {TRAINING_DATA_TYPE} data. Evaluating on Filtered data")
# x_data = np.copy(daily_per_mio_capita_filtered) 
# y_data = np.copy(daily_per_mio_capita_filtered) 

if daily_data == True:
    x_data, y_data = get_data_daily()
    y_data = np.copy(daily_per_mio_capita_filtered)
else:
    x_data, y_data = get_data_confirmed()
    y_data = np.copy(confirmed_per_mio_capita_filtered)

# y_data = np.copy(alert_unfilt) 

eval_model(model, x_data, y_data, dis=0, predict_data_label="Filtered")

# In[ ]:


x = 14
start_seqs = [np.random.random((1, x)) * 0,
              np.ones((1, x)) * 0,
              np.ones((1, x)) * 0.5,
              np.ones((1, x)) * 1,
              np.arange(x).reshape((1, x)) / 30,
              np.sin(np.arange(x) / x * np.pi / 2).reshape((1, x))
              ]

predictions = []
for start_seq in start_seqs:
    input_seq = np.copy(start_seq)
    predict_seq = [start_seq[0]]
    for _ in range(50):
        output = model(input_seq, training=False)
        predict_seq.append(output[0])
        input_seq = input_seq[:, output.shape[1]:]
        input_seq = np.append(input_seq, output).reshape((1, -1))

    predictions.append(np.concatenate(predict_seq))
plt.plot(np.array(predictions).T)
plt.show()

# In[ ]:


x = 21
start_seqs = [np.random.random((1, x, 1)) * 0,
              np.ones((1, x, 1)) * 0,
              np.ones((1, x, 1)) * 0.5,
              np.ones((1, x, 1)) * 1,
              np.arange(x).reshape((1, x, 1)) / 30,
              np.sin(np.arange(x) / x * np.pi / 2).reshape((1, x, 1))
              ]

predictions = []
for start_seq in start_seqs:
    input_seq = np.copy(start_seq)
    predict_seq = [start_seq[0, :, 0]]
    for _ in range(10):
        output = model(input_seq, training=False)
        predict_seq.append(output[0])
        input_seq = input_seq[:, output.shape[1]:, :]
        input_seq = np.append(input_seq, output).reshape((1, -1, 1))
    predictions.append(np.concatenate(predict_seq))
plt.plot(np.array(predictions).T)
plt.show()

# In[ ]:


# In[ ]:
