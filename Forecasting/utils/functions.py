import numpy  as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_dataset_random(x_data, y_data, x_size, y_size, n_samples, k_fold, k_fold_dimension, reduce_last_dim=False):
    """
    x_data : (regions, timesteps)
    y_data : (regions, timesteps)
    k_fold : k in k fold cross-validation
    k_fold_dimension : along which dimension to split in to k groups

    [___________] are from x_data
    [######]      are from y_data
    Note that the indexing is preserved when picking [________][#####] on x_data and y_data
                                                     10     20 21   25

    [___________][#####]
                [___________][#####]        
          [___________][#####]                                  
                                    [___________][#####]
                        [___________][#####]
    """
    
    n, t = x_data.shape
    
    if k_fold_dimension == 0: # split along region dimension. keep some regions seperate from training data
      # check if we can divide the data set to k folds
      samples_per_fold = n/k_fold
      if samples_per_fold < 1:
        raise Exception(f"Can't divide the dataset with {n} regions into {k_fold} folds")
      if samples_per_fold >= n:
        raise Exception(f"only 1 fold can be created. increase k")

      #randomize along region dimension consistantly with x_data and y_data
      index = np.arange(n)
      np.random.shuffle(index)
      x_data = x_data[index]
      y_data = y_data[index]

      # sererate into training and testing data
      selected_fold = np.random.randint(k_fold)
      """
      |   |   |    |    |     |    |
      [____________][___][__________]
                       ^  
                selected fold
      """
      a = selected_fold*samples_per_fold
      b = (selected_fold+1)*samples_per_fold
      x_data_train = np.concatenate([x_data[:a,:], x_data[b:,:]], 0)
      y_data_train = np.concatenate([y_data[:a,:], y_data[b:,:]], 0)
      x_data_test = x_data[a:b,:]
      y_data_test = y_data[a:b,:]
       
    elif k_fold_dimension == 1:# split along time dimension. keep some data from a time period seperate from training data
      # check if we can divide the data set to k folds
      samples_per_fold = t/k_fold
      if samples_per_fold < 1:
        raise Exception(f"Can't divide the dataset with {t} days into {k_fold} folds")
      if samples_per_fold >= n:
        raise Exception(f"only 1 fold can be created. increase k")
      if samples_per_fold < x_size + y_size:
        raise Exception(f"window size too large for the testing fold. no samples in testing fold. decrease window size or reduce k")

      # NOTE we cant randomize along time dimension. poop
      
      # we can't remove a fold from the middle of the time dimension and join 
      # the other two.
      
      # if we do that there will be a discontinuity
      
      # keep them  as it is and when we select random indexes we will see whether
      # it falls in to testing fold and put it accordingly
      x_data_train = x_data
      y_data_train = y_data
      x_data_test = x_data
      y_data_test = y_data
      
      selected_fold = np.random.randint(k_fold)
      a = selected_fold*samples_per_fold
      b = (selected_fold+1)*samples_per_fold
      
    x_train, y_train, x_test, y_test = [], [], [], []
    # create training data
    _N = 0
    while _N < n_samples*0.8:
        i = np.random.randint(0, t - x_size - y_size, 1)[0]
        if k_fold_dimension==1 and (a < i < b or a < i + x_size+y_size<b):
            continue
        x_train.append(x_data_train[:, i:i + x_size].T)
        y_train.append(y_data_train[:, i + x_size:i + x_size + y_size].T)
        _N += 1
    # create testing data
    _N = 0
    while _N < n_samples*0.2:
        i = np.random.randint(0, t - x_size - y_size, 1)[0]
        if k_fold_dimension==1 and (a < i < b or a < i + x_size+y_size<b):
            x_test.append(x_data_test[:,i:i+x_size].T)
            y_test.append(y_data_test[:, i+x_size:i+x_size+y_size].T)
            _N += 1

        if k_fold_dimension==0:
            x_test.append(x_data_test[:,i:i+x_size].T)
            y_test.append(y_data_test[:, i+x_size:i+x_size+y_size].T)        
            _N += 1
    X_train = np.stack(x_train, 0)
    Y_train = np.stack(y_train, 0)
    X_test = np.stack(x_test, 0)
    Y_test = np.stack(y_test, 0)
    if reduce_last_dim:
        X_train = np.concatenate(X_train,-1).T
        Y_train = np.concatenate(Y_train,-1).T
        X_test = np.concatenate(X_test,-1).T
        Y_test = np.concatenate(Y_test,-1).T
    return X_train, Y_train, X_test, Y_test

def split_into_pieces_random(x_data, y_data, x_size, y_size, N, reduce_last_dim=False):
    """
    x_data : (regions, timesteps)
    y_data : (regions, timesteps)

    [___________] are from x_data
    [######]      are from y_data
    Note that the indexing is preserved when picking [________][#####] on x_data and y_data
                                                     10     20 21   25

    [___________][#####]
                [___________][#####]        
          [___________][#####]                                  
                                    [___________][#####]
                        [___________][#####]
    """
    
    n, t = x_data.shape
    x, y = [], []
    
    _N = 0
    while _N != N:
        i = np.random.randint(0, t - x_size - y_size, 1)[0]
        if i + x_size + y_size > t:
            continue
        x.append(x_data[:, i:i + x_size].T)
        y.append(y_data[:, i + x_size:i + x_size + y_size].T)
        _N += 1
        
    X = np.stack(x, 0)
    Y = np.stack(y, 0)
    if reduce_last_dim:
        X = np.concatenate(X,-1).T
        Y = np.concatenate(Y,-1).T
    return X, Y

def split_into_pieces_inorder(x_data, y_data, x_size, y_size, window_size, reduce_last_dim=False):
    """
    x_data : (regions, timesteps)
    y_data : (regions, timesteps)

    [___________] are from x_data
    [######]      are from y_data
    Note that the indexing is preserved when picking [________][#####] on x_data and y_data
                                                     10     20 21   25

    [___________][#####]
    <--window_size-->[___________][#####]
                     <--window_size-->[___________][#####]
    """
    n, t = x_data.shape
    x, y = [], []
    
    for i in range(0, t, window_size):
        if i + x_size + y_size > t:
            continue
        x.append(x_data[:, i:i + x_size].T)
        y.append(y_data[:, i + x_size:i + x_size + y_size].T)
    X = np.stack(x, 0)
    Y = np.stack(y, 0)
    if reduce_last_dim:
        X = np.concatenate(X,-1).T
        Y = np.concatenate(Y,-1).T
    return X, Y

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

def convert_lon_lat_to_adjacency_matrix(df, lat_column="Lat", lon_column="Long", delta = 1e-5):
    N = len(df)
    A = np.zeros((N, N))
    i,j=0,0
    for _i, row_i in df.iterrows():
        j=0
        for _j, row_j in df.iterrows():
            A[i][j] = distance(row_i[lat_column], row_j[lat_column], row_i[lon_column], row_j[lon_column])
            if A[i][j] < delta:
                A[i][j] = 0
            if i==j:
                A[i][j] = 0
            
            if A[i][j] > 0:
                A[i][j] = 1/A[i][j]
            
            j += 1
        i += 1
    print("Adjacency matrix created")
    A = A/A.max()
    return A.astype(np.float32)

