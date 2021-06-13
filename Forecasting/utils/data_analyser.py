import math, random
import numpy as np
import scipy
import statsmodels.tsa.stattools as stattools


# check FFT of each region
def check_spectral(dataset, region_names, window):
    cases_fft = []
    for i in range(len(region_names)):
        fft_temp = np.abs(scipy.fft.fft(dataset[i, :]))[0:window]
        cases_fft.append(fft_temp)
    cases_fft = np.array(cases_fft)
    _mean = np.mean(cases_fft, axis=0)
    _var = np.var(cases_fft, axis=0)
    fft_mean = _mean / np.max(_mean)
    fft_var = _var / np.max(_var)

    return fft_mean, fft_var


# check ACF and PACF of each region
def check_acf(dataset, region_names, window):
    cases_acf = []
    cases_pacf = []
    for i in range(len(region_names)):
        _acf = stattools.acf(dataset[i, :], adjusted=True, nlags=window, fft=True, missing="drop")
        _pacf = stattools.pacf(dataset[i, :], nlags=window)
        cases_acf.append(_acf)
        cases_pacf.append(_pacf)
    cases_acf = np.array(cases_acf)
    cases_pacf = np.array(cases_pacf)
    acf_mean = np.mean(cases_acf, axis=0)
    pacf_mean = np.mean(cases_pacf, axis=0)

    return acf_mean, pacf_mean
