import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Forecasting.utils.data_loader import load_data_eu, load_data, load_smooth_data, load_samples, load_multiple_data
from Forecasting.utils.undersampling import undersample3, undersample_random


def main():
    # for country in countries:
    #     print('\nregion is: ' + country)
    fil, raw, fs = load_multiple_data(DATASETS=countries, data_path=dataset_path,
                                      look_back_window=look_back_window,
                                      window_slide=window_slide, R_EIG_ratio=R_EIG_ratio,
                                      R_power=R_power, midpoint=False)
    temp = load_samples(fil, fs, WINDOW_LENGTH, PREDICT_STEPS)
    x_train_list, y_train_list, x_test_list, y_test_list, x_val_list, y_val_list, fs_train, fs_test, fs_val = temp

    if optimised:
        x_train_uf, y_train_uf, x_train_feat = undersample3(x_train_list, y_train_list, fs_train, count_h, count_l,
                                                            num_h, num_l, power_l, power_h, power_penalty, clip,
                                                            clip_percentages, str(countries), plot_)
    else:
        x_train_uf, y_train_uf, x_train_feat = undersample_random(x_train_list, y_train_list, fs_train, ratio,
                                                                  str(countries), plot_)


# %% SCRIPT STARTS HERE

# data loading parameters
dataset_path = '../Datasets'
_df = pd.read_csv(os.path.join(dataset_path, "EU\jrc-covid-19-all-days-by-regions.csv"))
_eu = _df['CountryName'].unique().tolist()
# countries = ['SL']
# countries = ["ALL"]
countries = ['IT', 'SL', 'NG', 'Texas']
WINDOW_LENGTH = 14
PREDICT_STEPS = 7

# filtering parameters
midpoint = True
if midpoint:
    R_EIG_ratio = 1
    R_power = 2 / 3
else:
    R_EIG_ratio = 3
    R_power = 1

look_back_window = 100
window_slide = 1

# under-sampling parameters
optimise_ = [True]

for optimised in optimise_:
    clip = True
    plot_ = True

    if optimised:
        if clip:
            clip_percentages = [0, 10]
        count_h, count_l, num_h, num_l = 2, 0.2, 100000, 500
        power_l, power_h, power_penalty = 0.2, 2, 1000
    else:
        ratio = 0.3

    # >> MAIN LOOP
    if __name__ == "__main__":
        main()
