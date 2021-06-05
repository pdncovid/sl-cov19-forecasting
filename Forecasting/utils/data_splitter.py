import numpy as np
from Forecasting.utils.smoothing_functions import O_LPF, NO_LPF, O_NDA, NO_NDA


def split_and_smooth(x, look_back_window=100, window_slide=10, R_EIG_ratio=1, midpoint=False, reduce_last_dim=False):
    print(f"Split and smooth. Expected (nregions, days) Got {x.shape}. Look back window {look_back_window}")
    _x_to_smooth, _ = split_into_pieces_inorder(x, x, look_back_window, 0, window_slide, reduce_last_dim=False)
    _x = []
    for i in range(_x_to_smooth.shape[-1]):
        _x_samples_filtered, cutoff_freqs = O_LPF(_x_to_smooth[:, :, i], datatype='daily', order=3,R_EIG_ratio=R_EIG_ratio,
                                                  midpoint=midpoint, corr=True,
                                                  plot_freq=1, view=False,
                                                  region_names=[i for i in range(len(_x_to_smooth))])
        _x.append(_x_samples_filtered)
    _x = np.array(_x)
    _x = _x.transpose([1, 2, 0])
    if reduce_last_dim:
        _x = np.concatenate(_x, -1).T
    return _x, _x_to_smooth


def split_on_time_dimension(x_data, y_data, features, x_size, y_size, k_fold, test_fold, reduce_last_dim=False,
                            only_train_test=False, debug=False):
    """
    x_data : (n_regions, timesteps)
    y_data : (n_regions, timesteps)
    k_fold : k in k fold cross-validation
    """
    if k_fold < 3:
        raise Exception("k should be >=3")
    if test_fold < 2:
        raise Exception("test fold should be >=2")
    n, t = x_data.shape

    # check if we can divide the data set to k folds
    samples_per_fold = int(np.floor(t / k_fold))

    if samples_per_fold < 1:
        raise Exception(
            f"""Can't divide the dataset with {t} days into {k_fold} folds. Decrease k to a value less than {t / (
                    x_size + y_size)}""")
    # if samples_per_fold >= n:
    #   raise Exception(f"only 1 fold can be created. Increase k")
    if samples_per_fold < x_size + y_size:
        raise Exception(
            f"window size too large for the testing fold. no samples in testing fold. decrease window size or reduce k")

    # NOTE we cant randomize along time dimension.

    """
    |   |   |    |    |     |    |
    0        a   b    c           t
    [_______][___][___][__________]
          ^     ^    ^        ^    
      train  val   test   ignored 
                (test_fold)
    """

    # test_fold = np.random.randint(k_fold)
    a = (test_fold - 1) * samples_per_fold
    b = (test_fold - 0) * samples_per_fold
    c = (test_fold + 1) * samples_per_fold
    if only_train_test:
        a = b

    # keep them  as it is and when we select random indexes we will see whether
    # it falls in to testing fold and put it accordingly
    x_data_train = x_data[:, :a]
    y_data_train = y_data[:, :a]
    x_data_val = x_data[:, a:b]
    y_data_val = y_data[:, a:b]
    x_data_test = x_data[:, b:c]
    y_data_test = y_data[:, b:c]

    if debug:
        print()
        print("k_folds", k_fold)
        print("samples per fold", samples_per_fold)
        print(f"0, val start a={a}, test start b={b}, test end c={c}, {t}")
        print(
            f"""x_train:{x_data_train.shape} y_train:{y_data_train.shape} 
            x_val:{x_data_val.shape} y_val:{y_data_val.shape} 
            x_test:{x_data_test.shape} y_test:{y_data_test.shape}""")

    # create training data
    if debug:
        print(f"selecting {t - (1 if only_train_test else 2) * samples_per_fold} samples from training part")
    x_train, y_train = split_into_pieces_random(x_data_train, y_data_train, x_size, y_size,
                                                t - (1 if only_train_test else 2) * samples_per_fold, reduce_last_dim)
    # create testing data
    if debug:
        print(f"selecting {samples_per_fold} samples from testing part")
    x_test, y_test = split_into_pieces_random(x_data_test, x_data_test, x_size, y_size, samples_per_fold,
                                              reduce_last_dim)

    # create validation data
    if debug:
        print(f"selecting {samples_per_fold} samples from validation part")
    if only_train_test:
        x_val = np.zeros((0, *x_test.shape[1:]))
        y_val = np.zeros((0, *y_test.shape[1:]))
    else:
        x_val, y_val = split_into_pieces_random(x_data_val, x_data_val, x_size, y_size, samples_per_fold,
                                                reduce_last_dim)

    x_train_feat = np.expand_dims(features.T, 0).repeat(x_train.shape[0], 0)
    x_val_feat = np.expand_dims(features.T, 0).repeat(x_val.shape[0], 0)
    x_test_feat = np.expand_dims(features.T, 0).repeat(x_test.shape[0], 0)

    return x_train, x_train_feat, y_train, x_val, x_val_feat, y_val, x_test, x_test_feat, y_test


def split_on_region_dimension(x_data, y_data, x_size, y_size, n_samples, k_fold, test_fold, reduce_last_dim=False):
    """
    x_data : (n_regions, timesteps)
    y_data : (n_regions, timesteps)
    k_fold : k in k fold cross-validation
    """

    n, t = x_data.shape

    # check if we can divide the data set to k folds
    samples_per_fold = int(np.floor(n / k_fold))

    if samples_per_fold < 1:
        raise Exception(f"Can't divide the dataset with {n} regions into {k_fold} folds. Reduce k")
    if samples_per_fold >= n:
        raise Exception(f"only 1 fold can be created. Increase k")

    # #randomize along region dimension consistantly with x_data and y_data
    # index = np.arange(n)
    # np.random.shuffle(index)
    # x_data = x_data[index]
    # y_data = y_data[index]

    # sererate into training and testing data
    # test_fold = np.random.randint(k_fold)
    """

          -----> time
           0__________
          |
          |
    train |
          |
          |a__________
    val   |
          |b__________
    test  |
          |c__________
    train |
          |n__________

    """
    a = (test_fold - 1) * samples_per_fold
    b = (test_fold - 0) * samples_per_fold
    c = (test_fold + 1) * samples_per_fold

    x_data_train = np.concatenate([x_data[:a, :], x_data[c:, :]], 0)
    y_data_train = np.concatenate([y_data[:a, :], y_data[c:, :]], 0)
    x_data_val = x_data[a:b, :]
    y_data_val = y_data[a:b, :]
    x_data_test = x_data[b:c, :]
    y_data_test = y_data[b:c, :]

    print()
    print("k_folds", k_fold)
    print("samples per fold", samples_per_fold)
    print(f"0, val start a={a}, test start b={b}, test end c={c}, {n}")
    print(
        f"x_train:{x_data_train.shape} y_train:{y_data_train.shape} x_val:{x_data_val.shape} y_val:{y_data_val.shape} x_test:{x_data_test.shape} y_test:{y_data_test.shape}")

    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
    # create training data
    print(f"selecting {n_samples} samples from training part")
    _N = 0
    while _N < n_samples:
        i = np.random.randint(0, t - x_size - y_size, 1)[0]
        x_train.append(x_data_train[:, i:i + x_size].T)
        y_train.append(y_data_train[:, i + x_size:i + x_size + y_size].T)
        _N += 1
    # create validation data
    print(f"selecting {n_samples} samples from validation part")
    _N = 0
    while _N < n_samples:
        i = np.random.randint(0, t - x_size - y_size, 1)[0]
        x_val.append(x_data_val[:, i:i + x_size].T)
        y_val.append(y_data_val[:, i + x_size:i + x_size + y_size].T)
        _N += 1
    # create testing data
    print(f"selecting {n_samples} samples from testing part")
    _N = 0
    while _N < n_samples:
        i = np.random.randint(0, t - x_size - y_size, 1)[0]
        x_test.append(x_data_test[:, i:i + x_size].T)
        y_test.append(y_data_test[:, i + x_size:i + x_size + y_size].T)
        _N += 1

    X_train = np.stack(x_train, 0)
    Y_train = np.stack(y_train, 0)
    X_val = np.stack(x_val, 0)
    Y_val = np.stack(y_val, 0)
    X_test = np.stack(x_test, 0)
    Y_test = np.stack(y_test, 0)
    if reduce_last_dim:
        X_train = np.concatenate(X_train, -1).T
        Y_train = np.concatenate(Y_train, -1).T
        X_val = np.concatenate(X_val, -1).T
        Y_val = np.concatenate(Y_val, -1).T
        X_test = np.concatenate(X_test, -1).T
        Y_test = np.concatenate(Y_test, -1).T
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


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
        end = t - (x_size + y_size)
        i = 0 if end == 0 else np.random.randint(0, end)
        x.append(x_data[:, i:i + x_size].T)
        y.append(y_data[:, i + x_size:i + x_size + y_size].T)
        _N += 1

    X = np.stack(x, 0)
    Y = np.stack(y, 0)
    if reduce_last_dim:
        X = np.concatenate(X, -1).T
        Y = np.concatenate(Y, -1).T
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
        X = np.concatenate(X, -1).T
        Y = np.concatenate(Y, -1).T
    return X, Y
