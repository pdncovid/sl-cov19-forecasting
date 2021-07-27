import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stattools

from Forecasting.utils.data_loader import load_data_eu, load_data, load_smooth_data, load_samples, load_multiple_data, \
    show_curves
from Forecasting.utils.undersampling import undersample3, undersample_random
from Forecasting.utils.data_analyser import check_acf


# from Forecasting.models import get_model


def main():
    x_train = []
    y_train = []
    x_feat = []
    for countries in countries1:
        # data = load_data('Spain', path=dataset_path)
        # daily_cases = data['daily_cases']
        # region_names = data['region_names']
        # start_date = data['START_DATE']
        # print(region_names)
        # print(start_date)
        # print(daily_cases.shape)
        # plt.figure(figsize=(12, 4))
        # num = 1

        fil, raw, fs = load_multiple_data(DATASETS=countries, data_path=dataset_path,
                                          look_back_window=look_back_window,
                                          window_slide=window_slide, R_EIG_ratio=R_EIG_ratio,
                                          R_power=R_power, midpoint=midpoint)

        temp = load_samples(fil, fs, WINDOW_LENGTH, PREDICT_STEPS)

        x_train_list, y_train_list, x_test_list, y_test_list, x_val_list, y_val_list, fs_train, fs_test, fs_val = temp

        if optimised:
            x_train_uf, y_train_uf, x_train_feat = undersample3(x_train_list, y_train_list, fs_train, window_slide,
                                                                clip,
                                                                str(countries), plot_, repeat)
        else:
            x_train_uf, y_train_uf, x_train_feat = undersample_random(x_train_list, y_train_list, fs_train, ratio,
                                                                      str(countries), plot_)

        x_train_uf = np.array(x_train_uf)
        y_train_uf = np.array(y_train_uf)
        x_train_feat = np.array(x_train_feat)

        x_train.append(x_train_uf)
        y_train.append(y_train_uf)
        x_feat.append(x_train_feat)

    x_train_all = np.concatenate(x_train, axis=1)
    y_train_all = np.concatenate(y_train, axis=1)
    x_feat_all = np.concatenate(x_feat, axis=1)
    print(x_train_all.shape, y_train_all.shape, x_feat_all.shape)


# %% SCRIPT STARTS HERE

# data loading parameters

dataset_path = '../Datasets'
_df = pd.read_csv(os.path.join(dataset_path, "EU\jrc-covid-19-all-days-by-regions.csv"))
_eu = _df['CountryName'].unique().tolist()
# countries = ['Spain']
countries1 = ["Texas", "IT", "BD", "KZ", "KR", "Germany", "NG"]
countries1 = ['Norway']

WINDOW_LENGTH = 50
PREDICT_STEPS = 10
# filtering parameters

midpoint = True
if midpoint:
    R_EIG_ratio = 1.02
    R_power = 1
else:
    R_EIG_ratio = 3
    R_power = 1

look_back_window = 100
window_slide = 10
repeat = False
# window_slide = 1

# under-sampling parameters

optimised = True
clip = True
plot_ = True
sample_all = False
ratio = 0.3

# >> MAIN LOOP
if __name__ == "__main__":
    main()
