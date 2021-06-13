import numpy  as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Mean squared error with boosting low frequency samples
def bs(arr, v):
    l, r = 0, len(arr)
    while r - l > 0:
        i = (r + l) // 2
        if arr[i] < v:
            l = i + 1
        elif arr[i] > v:
            r = i
        else:
            return i
    return l


# ==================================== TODO: find a good normalization technique
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


def normalize_3d_xy_data(x_data, y_data):
    # input shape is [regions, samples, input sequence size] - list of regions
    for i in range(len(x_data)):
        shift = min(np.amin(x_data[i]), np.amin(y_data[i]))
        deno = max(np.amax(x_data[i]), np.amax(y_data[i])) - shift

        if deno == 0:
            deno = 1
        x_data[i] = (x_data[i] - shift) / deno
        y_data[i] = (y_data[i] - shift) / deno
    return x_data, y_data


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


def distance(lat1, lat2, lon1, lon2):
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    return c * r


def convert_lon_lat_to_adjacency_matrix(df, lat_column="Lat", lon_column="Long", delta=1e-5):
    N = len(df)
    A = np.zeros((N, N))
    i, j = 0, 0
    for _i, row_i in df.iterrows():
        j = 0
        for _j, row_j in df.iterrows():
            A[i][j] = distance(row_i[lat_column], row_j[lat_column], row_i[lon_column], row_j[lon_column])
            if A[i][j] < delta:
                A[i][j] = 0
            if i == j:
                A[i][j] = 0

            if A[i][j] > 0:
                A[i][j] = 1 / A[i][j]

            j += 1
        i += 1
    print("Adjacency matrix created")
    A = A / A.max()
    return A.astype(np.float32)
