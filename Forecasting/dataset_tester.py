import os

import numpy as np
import pandas as pd

from Forecasting.utils.data_loader import load_data_eu, load_data, load_smooth_data, load_samples,load_multiple_data
from Forecasting.utils.undersampling import undersample3


def main():
    # for country in countries:
    #     print('\nregion is: ' + country)

    daily_split_filtered, daily_split = load_multiple_data(DATASETS=countries, data_path=dataset_path,
                                                         look_back_window=look_back_window,
                                                         window_slide=window_slide, R_EIG_ratio=R_EIG_ratio,
                                                         R_power=R_power, midpoint=False)

    x_train, y_train, _, _, _, _ = load_samples(daily_split_filtered, WINDOW_LENGTH, PREDICT_STEPS)
    f_train = np.random.random((x_train.shape[0], 2, x_train.shape[2]))  # dummy features

    x_train_uf, y_train_uf, x_train_f = undersample3("ALL", x_train, y_train, f_train, count_h, count_l, num_h, num_l,
                                                     power_l, power_h, power_penalty, True)


# %% SCRIPT STARTS HERE

# data loading parameters
dataset_path = '../Datasets'
_df = pd.read_csv(os.path.join(dataset_path, "EU\jrc-covid-19-all-days-by-regions.csv"))
_eu = _df['CountryName'].unique().tolist()
countries = ['SL']
# countries = ['IT', 'SL', 'NG', 'Texas']
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
count_h, count_l, num_h, num_l = 2, 0.2, 45000, 500
power_l, power_h, power_penalty = 0.2, 2, 1000
# >> MAIN LOOP
if __name__ == "__main__":
    main()
