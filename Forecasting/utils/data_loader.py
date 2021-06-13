import pandas as pd
import numpy as np
import os, sys

from Forecasting.utils.data_splitter import split_and_smooth, split_on_time_dimension
from Forecasting.utils.functions import normalize_for_nn
from Forecasting.utils.smoothing_functions import O_LPF
import matplotlib.pyplot as plt


# normalise all as 0-1
def min_max(data):
    for k in range(data.shape[0]):
        data[k, :] = (data[k, :] - np.amin(data[k, :])) / (np.amax(data[k, :]) - np.amin(data[k, :]))
    return data


def reduce_regions_to_batch(arrs):
    # input arrs : (n_arrays, n_regions*, samples*, window)
    # output : (n_arrays, n_regions x samples, window)
    ret = []
    for arr in arrs:
        new_arr = []
        for i_regions in range(len(arr)):
            for i_sample in range(len(arr[i_regions])):
                new_arr.append(arr[i_regions][i_sample])
        ret.append(np.array(new_arr))
    return ret


def expand_dims(arrs, to):
    ret = []
    for arr in arrs:
        if len(arr.shape) != to:
            ret.append(np.expand_dims(arr, -1))
        else:
            ret.append(arr)
    return ret


def per_million(data, population):
    # divide by population
    data_per_mio_capita = np.zeros_like(data)
    for i in range(len(population)):
        data_per_mio_capita[i, :] = data[i, :] / population[i] * 1e6
    return data_per_mio_capita


def get_daily(data):
    # get daily cases by taking 1st order differencing
    data = np.diff(data)
    data[data < 0] = 0
    # # drop first day because daily_case arrays has 1 less data on time dimension
    # confirmed_cases = confirmed_cases[:,1:]
    # confirmed_filtered = confirmed_filtered[:,1:]
    return data


def get_data(filtered, normalize, data, dataf, population):
    if not filtered:
        x, y = np.copy(data), np.copy(data)
    else:
        x, y = np.copy(dataf), np.copy(dataf)

    x = per_million(x, population)
    y = per_million(y, population)
    if normalize:

        x, xs = normalize_for_nn(x, None if type(normalize) == bool else normalize)
        y, xs = normalize_for_nn(y, xs)
        return x.T, y.T, xs
    else:
        return x.T, y.T, None


def save_smooth_data(DATASET, data_path, look_back_window, window_slide, R_EIG_ratio, R_power,
                     midpoint):
    d = load_data(DATASET, path=data_path)
    region_names = d["region_names"]
    confirmed_cases = d["confirmed_cases"]
    daily_cases = d["daily_cases"]
    features = d["features"]
    daily_cases[daily_cases < 0] = 0
    population = features["Population"]

    daily_filtered, _ = O_LPF(daily_cases, datatype='daily', order=3, midpoint=midpoint, corr=True,
                              R_EIG_ratio=R_EIG_ratio, R_power=R_power, region_names=region_names,
                              plot_freq=1, view=False)
    # creates dataset
    _, _, x_data_scalers = get_data(False, normalize=True, data=daily_cases, dataf=daily_filtered,
                                    population=population)

    # dont get why this is there but ok
    x_data, y_data, _ = get_data(False, normalize=x_data_scalers, data=daily_cases, dataf=daily_filtered,
                                 population=population)  # get raw data

    # smooths data, now we have (samples x window x regions)
    _x, _x_to_smooth = split_and_smooth(x_data.T, look_back_window=look_back_window, window_slide=window_slide,
                                        R_EIG_ratio=R_EIG_ratio, R_power=R_power,
                                        midpoint=midpoint,
                                        reduce_last_dim=False)
    # tries something
    try:
        os.makedirs(f'./smoothed_data/{DATASET}')
    except FileExistsError as e:
        pass
    f_name = f"{look_back_window}_{window_slide}_{R_EIG_ratio}"
    np.save(f'./smoothed_data/{DATASET}/data_windows_{f_name}_smoothed', _x)
    np.save(f'./smoothed_data/{DATASET}/data_windows_{f_name}_original', _x_to_smooth)

    return _x, _x_to_smooth


def load_smooth_data(DATASET, data_path, look_back_window, window_slide, R_EIG_ratio, R_power,
                     midpoint):
    f_name = f"{look_back_window}_{window_slide}_{R_EIG_ratio}"
    from pathlib import Path
    my_file = Path(f'./smoothed_data/{DATASET}/data_windows_{f_name}_smoothed.npy')
    if my_file.is_file():
        print("Loading data from saved smoothed data folder ^_^")
        _x = np.load(f'./smoothed_data/{DATASET}/data_windows_{f_name}_smoothed.npy')
        _x_to_smooth = np.load(f'./smoothed_data/{DATASET}/data_windows_{f_name}_original.npy')
    else:
        print("Need to smooth! smoothing now ^_^ ")
        _x, _x_to_smooth = save_smooth_data(DATASET, data_path, look_back_window, window_slide, R_EIG_ratio, R_power,
                                            midpoint)

    return _x, _x_to_smooth


def load_multiple_data(DATASETS, data_path, look_back_window, window_slide, R_EIG_ratio, R_power,
                       midpoint):
    ret_smooth, ret_raw = [], []
    features = []
    print(f"Loading datasets {DATASETS}")
    if type(DATASETS) == str:
        DATASETS = DATASETS.split()
    for DATASET in DATASETS:

        print(f"Now Loading {DATASET} =)")
        tmp_smoothed, tmp_raw = load_smooth_data(DATASET, data_path, look_back_window, window_slide, R_EIG_ratio,
                                                 R_power,
                                                 midpoint)
        features.append(np.zeros((tmp_smoothed.shape[-1], 2)))  # dummy features

        # tmp_smoothed = reduce_regions_to_batch([tmp_smoothed])[0]
        # tmp_smoothed = expand_dims([tmp_smoothed], 3)[0]
        ret_smooth.append(tmp_smoothed)

        # tmp_raw = reduce_regions_to_batch([tmp_raw])[0]
        # tmp_raw = expand_dims([tmp_raw], 3)[0]
        ret_raw.append(tmp_raw)

    #  if the shape of ret_raw : (datasets, samples*, lookback, regions*)
    ret_smooth2, ret_raw2, features2 = [], [], []
    for i in range(len(DATASETS)):
        for j in range(ret_smooth[i].shape[-1]):  # n_regions will be different for each dataset
            ret_smooth2.append(ret_smooth[i][:, :, j])
            ret_raw2.append(ret_raw[i][:, :, j])
            features2.append(features[i][j])

    # # if the shape of ret_raw : (datasets, samples*, lookback)
    # ret_raw2 = np.zeros((0, look_back_window, 1))
    # ret_smooth2 = np.zeros((0, look_back_window, 1))
    # for i in range(len(DATASETS)):
    #     ret_raw2 = np.concatenate([ret_raw2, np.array(ret_raw[i])], 0)
    #     ret_smooth2 = np.concatenate([ret_smooth2, np.array(ret_smooth[i])], 0)
    return ret_smooth2, ret_raw2, features2


def load_samples(_x, fs, WINDOW_LENGTH, PREDICT_STEPS):
    # input _x : (regions, samples*, seqlength)
    # input fs : (regions, features*)
    x_train_list, y_train_list, x_test_list, y_test_list, x_val_list, y_val_list = [], [], [], [], [], []
    fs_train, fs_test, fs_val =[], [], []
    for i_region in range(len(_x)):
        x = _x[i_region][:, -WINDOW_LENGTH - PREDICT_STEPS:-PREDICT_STEPS]
        y = _x[i_region][:, -PREDICT_STEPS:]
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x_train_idx = np.ceil(len(idx) * 0.8).astype(int)
        x_train_list.append(x[idx[:x_train_idx]])
        y_train_list.append(y[idx[:x_train_idx]])
        x_test_list.append(x[idx[x_train_idx:]])
        y_test_list.append(y[idx[x_train_idx:]])
        x_val_list.append(np.zeros((0, *x_train_list[-1].shape[1:])))
        y_val_list.append(np.zeros((0, *y_train_list[-1].shape[1:])))

        fs_train.append(np.repeat(fs[i_region:i_region+1], x_train_list[-1].shape[0],0))
        fs_test.append(np.repeat(fs[i_region:i_region+1], x_test_list[-1].shape[0],0))
        fs_val.append(np.repeat(fs[i_region:i_region+1], x_val_list[-1].shape[0],0))

    return x_train_list, y_train_list, x_test_list, y_test_list, x_val_list, y_val_list, fs_train, fs_test, fs_val


# def save_train_data(DATASET, data_path, TRAINING_DATA_TYPE, WINDOW_LENGTH, PREDICT_STEPS, midpoint, R_EIG_ratio,
#                     R_power,
#                     look_back_filter, look_back_window, window_slide):
#     d = load_data(DATASET, path=data_path)
#     region_names = d["region_names"]
#     confirmed_cases = d["confirmed_cases"]
#     daily_cases = d["daily_cases"]
#     features = d["features"]
#     START_DATE = d["START_DATE"]
#     n_regions = d["n_regions"]
#     daily_cases[daily_cases < 0] = 0
#     population = features["Population"]
#     days = confirmed_cases.shape[1]
#     features = features.values
#
#     daily_filtered, cutoff_freqs = O_LPF(daily_cases, datatype='daily', order=3, midpoint=midpoint, corr=True,
#                                          R_EIG_ratio=R_EIG_ratio, R_power=R_power, region_names=region_names,
#                                          plot_freq=1, view=False)
#     x_data, y_data, x_data_scalers = get_data(False, normalize=True, data=daily_cases, dataf=daily_filtered,
#                                               population=population)
#     x_dataf, y_dataf, x_data_scalersf = get_data(True, normalize=True, data=daily_cases, dataf=daily_filtered,
#                                                  population=population)
#
#     print(f"Using {TRAINING_DATA_TYPE} data")
#     if look_back_filter and TRAINING_DATA_TYPE == "Filtered":
#         x_data, y_data, _ = get_data(False, normalize=x_data_scalers, data=daily_cases, dataf=daily_filtered,
#                                      population=population)  # get raw data
#
#         # smooth data
#         _x, _ = split_and_smooth(x_data.T, look_back_window=look_back_window, window_slide=window_slide,
#                                  R_EIG_ratio=R_EIG_ratio, R_power=R_power,
#                                  midpoint=midpoint,
#                                  reduce_last_dim=False)
#         X = _x[:, -WINDOW_LENGTH - PREDICT_STEPS:-PREDICT_STEPS, :]
#         Y = _x[:, -PREDICT_STEPS:, :]
#         idx = np.arange(X.shape[0])
#         np.random.shuffle(idx)
#         X_train_idx = np.ceil(len(idx) * 0.8).astype(int)
#         X_train = X[idx[:X_train_idx]]
#         Y_train = Y[idx[:X_train_idx]]
#         X_test = X[idx[X_train_idx:]]
#         Y_test = Y[idx[X_train_idx:]]
#         X_train_feat = np.expand_dims(features.T, 0).repeat(X_train.shape[0], 0)
#         X_test_feat = np.expand_dims(features.T, 0).repeat(X_test.shape[0], 0)
#
#         X_val = np.zeros((0, *X_train.shape[1:]))
#         Y_val = np.zeros((0, *Y_train.shape[1:]))
#         X_val_feat = np.zeros((0, *X_train_feat.shape[1:]))
#
#
#     else:
#         x_data, y_data, _ = get_data(TRAINING_DATA_TYPE == "Filtered", normalize=x_data_scalers, data=daily_cases,
#                                      dataf=daily_filtered, population=population)
#         X_train, X_train_feat, Y_train, X_val, X_val_feat, Y_val, X_test, X_test_feat, Y_test = split_on_time_dimension(
#             x_data.T, y_data.T, features, WINDOW_LENGTH, PREDICT_STEPS,
#             k_fold=3, test_fold=2, reduce_last_dim=False,
#             only_train_test=True, debug=True)
#
#     if len(X_train.shape) == 2:
#         X_train, X_train_feat, Y_train = np.expand_dims(X_train, -1), np.expand_dims(X_train_feat, -1), np.expand_dims(
#             Y_train, -1)
#         X_val, X_val_feat, Y_val = np.expand_dims(X_val, -1), np.expand_dims(X_val_feat, -1), np.expand_dims(Y_val, -1)
#         X_test, X_test_feat, Y_test = np.expand_dims(X_test, -1), np.expand_dims(X_test_feat, -1), np.expand_dims(
#             Y_test, -1)
#
#     X_train = np.concatenate([X_train, X_val], 0)
#     X_train_feat = np.concatenate([X_train_feat, X_val_feat], 0)
#     Y_train = np.concatenate([Y_train, Y_val], 0)
#     try:
#         os.makedirs(f'./preprocessed_data/{DATASET}')
#     except FileExistsError as e:
#         pass
#     fname = f"{TRAINING_DATA_TYPE}_{WINDOW_LENGTH}_{PREDICT_STEPS}_{R_EIG_ratio}_{R_power}"
#     if look_back_filter and TRAINING_DATA_TYPE == "Filtered":
#         fname += f"_{midpoint}_{look_back_window}_{window_slide}"
#
#     np.save(f'./preprocessed_data/{DATASET}/X_train_{fname}', X_train)
#     np.save(f'./preprocessed_data/{DATASET}/Y_train_{fname}', Y_train)
#     np.save(f'./preprocessed_data/{DATASET}/X_train_feat_{fname}', X_train_feat)
#     np.save(f'./preprocessed_data/{DATASET}/X_test_{fname}', X_test)
#     np.save(f'./preprocessed_data/{DATASET}/Y_test_{fname}', Y_test)
#     np.save(f'./preprocessed_data/{DATASET}/X_test_feat_{fname}', X_test_feat)
#     np.save(f'./preprocessed_data/{DATASET}/X_val_{fname}', X_val)
#     np.save(f'./preprocessed_data/{DATASET}/Y_val_{fname}', Y_val)
#     np.save(f'./preprocessed_data/{DATASET}/X_val_feat_{fname}', X_val_feat)
#
#     fig, axs = plt.subplots(2, 2)
#     x_data, y_data, _ = get_data(filtered=False, normalize=True, data=daily_cases, dataf=daily_filtered,
#                                  population=population)
#     axs[0, 0].plot(x_data)
#     axs[0, 0].set_title("Original data")
#
#     for i in range(X_train.shape[-1]):
#         axs[0, 1].plot(np.concatenate([X_train[:, :, i], Y_train[:, :, i]], 1).T)
#     axs[0, 1].axvline(X_train.shape[1], color='r', linestyle='--')
#
#     axs[1, 0].hist(X_train.reshape(-1), bins=100)
#     axs[1, 0].set_title("Histogram of cases")
#
#     axs[1, 1].hist(np.concatenate(X_train, -1).mean(0), bins=100)
#     axs[1, 1].set_title("Histogram of mean of training samples")
#
#     plt.savefig(f'./preprocessed_data/{DATASET}/Train_data_{fname}.png', bbox_inches='tight')
#
#     return X_train, Y_train, X_train_feat, X_test, Y_test, X_test_feat, X_val, Y_val, X_val_feat


# def load_multiple_train_data(DATASETS, data_path, TRAINING_DATA_TYPE, WINDOW_LENGTH, PREDICT_STEPS, midpoint,
#                              R_EIG_ratio, R_power,
#                              look_back_filter, look_back_window, window_slide):
#     ret = []
#     for DATASET in DATASETS:
#         tmp = load_train_data(DATASET, data_path, TRAINING_DATA_TYPE, WINDOW_LENGTH, PREDICT_STEPS, midpoint,
#                               R_EIG_ratio, R_power,
#                               look_back_filter, look_back_window, window_slide)
#
#         if len(DATASETS) > 1:
#             tmp = reduce_regions_to_batch(tmp)  # can't load multiple datasets if we keep regions in separate dim
#             tmp = expand_dims(tmp, 3)
#
#         if len(ret) == 0:
#             ret = tmp
#         else:
#             for i in range(len(ret)):
#                 ret[i] = np.concatenate([ret[i], tmp[i]], 0)
#     return ret


# def load_train_data(DATASET, data_path, TRAINING_DATA_TYPE, WINDOW_LENGTH, PREDICT_STEPS, midpoint, R_EIG_ratio,
#                     R_power,
#                     look_back_filter, look_back_window, window_slide):
#     fname = f"{TRAINING_DATA_TYPE}_{WINDOW_LENGTH}_{PREDICT_STEPS}_{R_EIG_ratio}_{R_power}"
#     if look_back_filter and TRAINING_DATA_TYPE == "Filtered":
#         fname += f"_{midpoint}_{look_back_window}_{window_slide}"
#
#     from pathlib import Path
#
#     my_file = Path(f'./preprocessed_data/{DATASET}/X_train_{fname}.npy')
#     if my_file.is_file():
#         print("Loading data from saved preprocessed data folder...")
#         X_train = np.load(f'./preprocessed_data/{DATASET}/X_train_{fname}.npy')
#         Y_train = np.load(f'./preprocessed_data/{DATASET}/Y_train_{fname}.npy')
#         X_train_feat = np.load(f'./preprocessed_data/{DATASET}/X_train_feat_{fname}.npy')
#         X_test = np.load(f'./preprocessed_data/{DATASET}/X_test_{fname}.npy')
#         Y_test = np.load(f'./preprocessed_data/{DATASET}/Y_test_{fname}.npy')
#         X_test_feat = np.load(f'./preprocessed_data/{DATASET}/X_test_feat_{fname}.npy')
#         X_val = np.load(f'./preprocessed_data/{DATASET}/X_val_{fname}.npy')
#         Y_val = np.load(f'./preprocessed_data/{DATASET}/Y_val_{fname}.npy')
#         X_val_feat = np.load(f'./preprocessed_data/{DATASET}/X_val_feat_{fname}.npy')
#     else:
#         tmp = save_train_data(DATASET, data_path, TRAINING_DATA_TYPE, WINDOW_LENGTH, PREDICT_STEPS, midpoint,
#                               R_EIG_ratio, R_power,
#                               look_back_filter, look_back_window, window_slide)
#         X_train, Y_train, X_train_feat, X_test, Y_test, X_test_feat, X_val, Y_val, X_val_feat = tmp
#     return X_train, Y_train, X_train_feat, X_test, Y_test, X_test_feat, X_val, Y_val, X_val_feat


def load_data(DATASET, path="/content/drive/Shareddrives/covid.eng.pdn.ac.lk/COVID-AI (PG)/spatio_temporal/Datasets"):
    if DATASET == "SL":
        dataset_path = os.path.join(path, "SL")

        df_confirmed = pd.read_csv(os.path.join(dataset_path, "SL_covid_all_updated.csv"))
        df_confirmed = df_confirmed.set_index("Code").sort_index()
        df_population = pd.read_csv(os.path.join(dataset_path, "SL_population_updated.csv"))
        df_population = df_population.set_index("Code").sort_index()

        df_food_ratios = pd.read_csv(os.path.join(dataset_path, "foodexpenditure_ratios_updated.csv"))
        df_food_ratios = df_food_ratios.set_index("Code").sort_index()

        df_unemployment = pd.read_csv(os.path.join(dataset_path, "unemployment_updated.csv"))
        df_unemployment = df_unemployment.set_index("Code").sort_index()

        df_poverty = pd.read_csv(os.path.join(dataset_path, "povery_rates2012_updated.csv"))
        df_poverty = df_poverty.set_index("Code").sort_index()

        df_internet = pd.read_csv(os.path.join(dataset_path, "internet_percent_women_updated.csv"))
        df_internet = df_internet.set_index("Code").sort_index()

        df_education = pd.read_csv(os.path.join(dataset_path, "education_years_women_updated.csv"))
        df_education = df_education.set_index("Code").sort_index()

        df_industry = pd.read_csv(os.path.join(dataset_path, "industry_women_updated.csv"))
        df_industry = df_industry.set_index("Code").sort_index()

        # Dropping Kalmunai
        df_confirmed = df_confirmed.loc[df_confirmed['District'] != 'KALMUNAI']
        df_confirmed = df_confirmed.rename(columns={"District": "Region"})

        region_codes = df_confirmed.index

        confirmed_cases = np.array(np.float64(df_confirmed.iloc[0:25, 1:].values))
        daily_cases = np.diff(confirmed_cases)

        region_names = list(df_confirmed['Region'])
        region_names[25:] = []

        population = 1000 * df_population.iloc[:, 7]
        lat = df_population["Lat"]
        lon = df_population["Lon"]
        land = df_population["Land Area"]
        pop_density = pd.Series(population / land, name="Population density")
        labour_total = df_industry["Total percentage of persons involved in labour"]
        labour_skilled = df_industry["Percentage of persons involved in skilled labour"]
        labour_unskilled = df_industry["Percentage of persons involved in unskilled labour"]
        labour_agri = df_industry["Percentage of persons involved in agriculture"]

        unemployment = df_unemployment["Unemployment rate"]
        poverty = df_poverty["Poverty rate"]

        spending_total = df_food_ratios["Total monthly expenditure"]
        spending_food = df_food_ratios["Monthly expenditure on food and drink"]
        spending_other = df_food_ratios["Monthly expenditure on non-food items"]
        spending_ratio = df_food_ratios["Ratio between expenditure on food and non-food items"]

        internet = df_internet["Percentage of persons using internet"]
        education = df_education["Median years spent in education"]

        features = pd.concat(
            [population, lat, lon, pop_density, labour_total, labour_skilled, labour_unskilled, labour_agri,
             unemployment, poverty, spending_total, spending_food, spending_other, spending_ratio, internet, education],
            axis=1, join="inner").rename(
            columns={"Total (2017)": "Population",
                     "Total percentage of persons involved in labour": "Labour(Total %)",
                     "Percentage of persons involved in skilled labour": "Labour(Skilled %)",
                     "Percentage of persons involved in unskilled labour": "Labour(Unskilled %)",
                     "Percentage of persons involved in agriculture": "Labour(Agri %)",
                     "Total monthly expenditure": "Expenses(Total)",
                     "Monthly expenditure on food and drink": "Expenses(Food)",
                     "Monthly expenditure on non-food items": "Expenses(Non-food)",
                     "Ratio between expenditure on food and non-food items": "Expenses(Food/Non-food)",
                     "Percentage of persons using internet": "Internet usage(%)",
                     "Median years spent in education": "Education(Median years)"})

        START_DATE = "14/11/2020"

    elif DATASET == "Texas":
        START_DATE = "03/04/2020"

        dataset_path = os.path.join(path, "Texas")

        # dataframes
        df_confirmed = pd.read_csv(os.path.join(dataset_path, "Texas COVID-19 Case Count Data by County.csv"),
                                   skiprows=2, nrows=254)  # https://dshs.texas.gov/coronavirus/AdditionalData.aspx
        df_population = pd.read_csv(os.path.join(dataset_path, "2019_txpopest_county.csv"),
                                    header=0)  # https://demographics.texas.gov/Resources/TPEPP/Estimates/2019/2019_txpopest_county.csv

        # conv to np.array
        confirmed_cases = np.array(np.float64(df_confirmed.iloc[:, 1:].values))
        region_names = np.array(df_population.iloc[:-1, 1].values)

        population = df_population.iloc[:-1, 2]
        features = pd.concat([population], axis=1, join="inner").rename(columns={'cqr_census_2010_count': 'Population'})

        # fixing the confirmed cases dataset (negative gradients)
        for k in range(confirmed_cases.shape[0]):
            for i in range(confirmed_cases.shape[1] - 1):
                if confirmed_cases[k, i + 1] < confirmed_cases[k, i]:
                    confirmed_cases[k, i + 1] = confirmed_cases[k, i]

        daily_cases = np.diff(confirmed_cases, axis=-1)

        n_regions = confirmed_cases.shape[0]
        days = confirmed_cases.shape[1]

    elif DATASET == "USA":
        dataset_path = os.path.join(path, "US")

        state_names = pd.read_csv(os.path.join(dataset_path, "state_name.csv"), header=None)
        state_names.columns = ['State Name', 'State Code']

        df_daily = pd.read_csv(os.path.join(dataset_path, "cases_new.csv"), header=None)
        n_regions = len(state_names)
        region_names = [state_names.iloc[i, 0] for i in range(n_regions)]
        daily_cases = np.array(np.float64(df_daily.iloc[:, :].values))
        confirmed_cases = np.cumsum(daily_cases, axis=1)

        # features
        health = pd.read_csv(os.path.join(dataset_path, "healthins.csv"), header=None)
        povert = pd.read_csv(os.path.join(dataset_path, "poverty.csv"), header=None)
        income = pd.read_csv(os.path.join(dataset_path, "income.csv"), header=None)
        popden = pd.read_csv(os.path.join(dataset_path, "pop_density.csv"), header=None)
        population = pd.read_csv(os.path.join(dataset_path, "pop.csv"), header=None)
        features = pd.concat([population, popden, health, income, povert], axis=1)

        START_DATE = "14/01/2020"  # TODO FIND

    elif DATASET == "NG":
        dataset_path = os.path.join(path, "NG")
        df_daily = pd.read_excel(os.path.join(dataset_path, "nga_subnational_covid19_hera.xls"))
        df_daily = df_daily[['DATE', 'REGION', 'CONTAMINES']]
        dates = df_daily['DATE'].unique()
        region_names = df_daily['REGION'].unique()
        df_time = pd.DataFrame(columns=dates, index=region_names)

        for date in dates:
            df_date = df_daily.loc[df_daily['DATE'] == date]
            df_date = df_date[['REGION', 'CONTAMINES']]
            df_date = df_date.set_index('REGION')
            df_time.loc[df_date.index, date] = df_date.values.reshape(-1)

        # removing nan rows
        df_time = df_time[df_time.index.notnull()]

        # remove unspecified rows
        remove_rows = ['', ' ', 'nan', 'Nan', 'NOT SPECIFIED', 'Non spécifié']
        for word in remove_rows:
            if word in df_time.index:
                df_time = df_time.drop(index=word)
        # fill nan values
        df_time = df_time.fillna(value=0)
        daily_cases = np.array(np.float64(df_time.values))
        confirmed_cases = np.cumsum(daily_cases, axis=1)

        region_names = df_time.index
        features = pd.DataFrame(columns=['Population'], index=region_names)
        features['Population'] = 1e6

        START_DATE = df_time.columns[0]

    elif DATASET == "Global":
        dataset_path = os.path.join(path, "Global")

    elif DATASET == "IT":
        d = load_data_eu("Italy", True, path)
        region_names = d["region_names"]
        confirmed_cases = d["confirmed_cases"]
        daily_cases = d["daily_cases"]
        START_DATE = d["START_DATE"]
        n_regions = d["n_regions"]
    else:
        raise Exception(f"Dataset name {DATASET} not found!")
    features = pd.DataFrame(columns=['Population'], index=region_names)  # todo population ignored, features ignored now
    features['Population'] = 1e6
    daily_cases[daily_cases < 0] = 0
    return {
        "region_names": region_names,
        "confirmed_cases": confirmed_cases,
        "daily_cases": daily_cases,
        "features": features,
        "START_DATE": START_DATE,
        "n_regions": len(region_names),
    }


def load_data_eu(country='Germany', provinces=True,
                 path="/content/drive/Shareddrives/covid.eng.pdn.ac.lk/COVID-AI (PG)/spatio_temporal/Datasets"):
    dataset_path = os.path.join(path, "EU")

    if country != 'Italy':
        _df = pd.read_csv(os.path.join(dataset_path, "jrc-covid-19-all-days-by-regions.csv"))
        region_idx = _df.index[_df['CountryName'].str.contains(country)].tolist()
        column_name = 'Region'
        date_name = 'Date'
        covid_name = 'CumulativePositive'
    else:
        _df = pd.read_csv(os.path.join(dataset_path, "dpc-covid19-ita-province.csv"))
        region_idx = _df.index.tolist()
        if provinces:
            column_name = 'denominazione_provincia'
        else:
            column_name = 'denominazione_regione'
        date_name = 'data'
        covid_name = 'totale_casi'

    df_new = _df.iloc[region_idx, :][[date_name, column_name, covid_name]]

    region_list = _df[column_name].unique().tolist()
    dates = df_new[date_name].unique().tolist()

    df_time = pd.DataFrame(index=region_list, columns=dates)

    for _date in dates:
        _df = df_new.loc[df_new[date_name] == _date][[column_name, covid_name]]
        _df = _df.set_index(column_name)
        df_time.loc[_df.index, _date] = _df.values.reshape(-1)
        df_time[df_time.isnull().values] = 0

    # removing nan rows
    df_time = df_time[df_time.index.notnull()]
    # remove unspecified rows
    remove_rows = ['', ' ', 'nan', 'Nan', 'NOT SPECIFIED', 'Non spécifié']
    for word in remove_rows:
        if word in df_time.index:
            df_time = df_time.drop(index=word)
    # fill nan values
    df_time = df_time.fillna(value=0)

    confirmed_cases = df_time.values.astype(np.float64)

    for i in range(confirmed_cases.shape[0]):
        for j in range(confirmed_cases.shape[1] - 1):
            if confirmed_cases[i, j + 1] < confirmed_cases[i, j]:
                confirmed_cases[i, j + 1] = confirmed_cases[i, j]
    daily_cases = np.diff(confirmed_cases, axis=1).astype(np.float64)
    daily_cases[daily_cases < 0] = 0
    start_date = df_time.columns[0]

    return {
        "region_names": df_time.index,
        "confirmed_cases": confirmed_cases,
        "daily_cases": daily_cases,
        "START_DATE": start_date,
        "n_regions": len(df_time.index),
    }
