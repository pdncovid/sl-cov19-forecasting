# It is also very important to include some naive forecast as the series mean or previous value to make sure our models
# perform better than the simplest of the simplest. We dont want to introduce any complexity if it does not provides any
# performance gain. (*harshana I dont understand this statement fully*)

import numpy as np


def naive_mean(df_test):
    mean = df_test.mean()
    mean = np.array([mean for _ in range(len(df_test))])
    return mean


def naive_yesterday(df_test):
    return np.insert(df_test.values[:-1, :], 0, 0, 0)
