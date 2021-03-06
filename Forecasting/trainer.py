# -*- coding: utf-8 -*-

# from google.colab import drive
# drive.mount('/content/drive')
# import sys
# import os
# path = "/content/drive/Shareddrives/covid.eng.pdn.ac.lk drive/COVID-AI (PG)/spatio_temporal/sl-cov19-forecasting/Forecasting"
# os.chdir(path)
# sys.path.insert(0, os.path.join(sys.path[0], '..'))
import argparse
import os
import sys
import time
import matplotlib as mpl

mpl.use('Agg')

sys.path.insert(0, os.path.join(sys.path[0], '..'))
import pandas as pd  # Basic library for all of our dataset operations
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# plots

import matplotlib.pyplot as plt
from utils.plots import plot_prediction
from utils.functions import normalize_for_nn, undo_normalization, bs
from utils.data_loader import load_data, per_million, get_data, reduce_regions_to_batch, expand_dims, \
    load_multiple_data, load_samples
from utils.smoothing_functions import O_LPF
from utils.data_splitter import split_on_time_dimension, split_into_pieces_inorder, \
    split_and_smooth
from utils.undersampling import undersample3
from models import get_model

# Extra settings
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'
mpl.rcParams['figure.figsize'] = 18, 8

print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def get_loss_f(undersampling, xcheck, freq):
    def loss_f_normal(y_true, y_pred, x):
        y_pred = tf.dtypes.cast(y_pred, tf.float64)
        return tf.reduce_mean((y_true - y_pred) ** 2)

    def loss_f_new(y_true, y_pred, x):
        region_sample_freq = np.zeros(y_true.shape, dtype='double')

        for batch in range(y_true.shape[0]):
            for t in range(y_true.shape[1]):
                for n in range(y_true.shape[2]):
                    i = bs(xcheck, np.mean(x[batch, t, n])) - 1
                    region_sample_freq[batch, t, n] = freq[i]
        y_pred = tf.dtypes.cast(y_pred, tf.float64)
        se = (y_true - y_pred) ** 2

        return tf.reduce_mean(se * (1 / np.log(region_sample_freq)) ** 2 * 10)

    if undersampling == "Reduce" or undersampling == 'None':
        return loss_f_normal
    else:
        return loss_f_new


def eval_metric(y_true, y_pred):
    return np.mean((np.squeeze(y_true) - np.squeeze(y_pred)) ** 2) ** 0.5


def train(model, train_data, X_train, Y_train, X_test, Y_test):
    print("Model Input shape", model.input.shape)
    print("Model Output shape", model.output.shape)

    tensorboard = TensorBoard(log_dir='./logs/' + folder, write_graph=True, histogram_freq=1, write_images=True)
    tensorboard.set_model(model)

    opt = tf.keras.optimizers.Adam(lr=lr)
    loss_f = get_loss_f(UNDERSAMPLING, xcheck, freq)

    train_metric = []
    val_metric = []
    test_metric = []
    best_test_value = 1e10
    for epoch in range(EPOCHS):
        losses = []
        for x, y in train_data:
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_f(y, y_pred, x)

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
            print(f" Best test metric {best_test_value:.5f}. Saving model...")
            model.save("temp.h5")
        if PLOT:
            test1(model, str(epoch))
            plt.clf()

            plt.figure(16, figsize=(10, 3))
            plt.plot(train_metric, label='Train')
            plt.plot(test_metric, label='Test')
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.legend()
            plt.savefig(f"./logs/{folder}/images/Train_metric.png", bbox_inches='tight')
            plt.clf()

    model = tf.keras.models.load_model("temp.h5")
    model.save("models/" + fmodel_name + ".h5")


def main():
    # ============================================================================================ Initialize parameters
    parser = argparse.ArgumentParser(description='Train NN model for forecasting COVID-19 pandemic')
    parser.add_argument('--daily', help='Use daily data', action='store_true')
    parser.add_argument('--dataset', help='Dataset used for training. (SL, Texas, USA, Global)', type=str,
                        nargs='+', default="SL Texas NG IT BD KZ KR DEU")
    parser.add_argument('--test_days', help='number of days used for testing.', type=int, default=30)
    # parser.add_argument('--split_date', help='Train-Test splitting date', type=str, default='2021-4-01')

    parser.add_argument('--epochs', help='Epochs to be trained', type=int, default=50)
    parser.add_argument('--batchsize', help='Batch size', type=int, default=16)
    parser.add_argument('--input_days', help='Number of days input into the NN', type=int, default=50)
    parser.add_argument('--output_days', help='Number of days predicted by the model', type=int, default=10)
    parser.add_argument('--modeltype', help='Model type', type=str, default='LSTM_Simple_WO_Regions')

    parser.add_argument('--lr', help='Learning rate', type=float, default=0.004)
    parser.add_argument('--preprocessing', help='Preprocessing on the training data (Unfiltered, Filtered)', type=str,
                        default="Filtered")
    parser.add_argument('--undersampling', help='under-sampling method (None, Loss, Reduce)', type=str,
                        default="Loss")

    parser.add_argument('--path', help='default dataset path', type=str, default="../Datasets")
    parser.add_argument('--window_slide', help='window_slide', type=int, default=10)
    parser.add_argument('--load_recent', help='Use daily data', action='store_true')

    args = parser.parse_args()

    global daily_data, TEST_DAYS, DATASETS, split_date, EPOCHS, BATCH_SIZE, BUFFER_SIZE, WINDOW_LENGTH, PREDICT_STEPS, lr, \
        TRAINING_DATA_TYPE, UNDERSAMPLING, PLOT, test_daily_cases, test_daily_filtered, test_population, test_region_names, test_days_split_idx, \
        x_test_data_scalers, folder, fmodel_name, count_h, count_l, num_l, num_h, power_l, power_h, power_penalty, clip_percentages

    daily_data = args.daily
    DATASETS = args.dataset
    if type(DATASETS) == str:
        DATASETS = [DATASETS]
    if len(DATASETS) == 1:
        DATASETS = DATASETS[0].split(' ')
    TEST_DAYS = args.test_days
    # split_date = args.split_datte

    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    BUFFER_SIZE = 100
    WINDOW_LENGTH = args.input_days
    PREDICT_STEPS = args.output_days
    lr = args.lr
    TRAINING_DATA_TYPE = args.preprocessing
    UNDERSAMPLING = args.undersampling

    split_train_data = True
    midpoint = True

    if midpoint:
        R_EIG_ratio = 1.02
        R_power = 1
    else:
        R_EIG_ratio = 3
        R_power = 1

    look_back_window, window_slide = 100, args.window_slide
    PLOT = True

    # ================================================================================================ Loading test data
    d = load_data('SL', path=args.path)
    test_region_names = d["region_names"]
    test_daily_cases = d["daily_cases"]
    test_features = d["features"]
    test_population = test_features["Population"]
    test_daily_cases[test_daily_cases < 0] = 0
    test_daily_filtered, cutoff_freqs = O_LPF(test_daily_cases, datatype='daily', order=3, midpoint=midpoint, corr=True,
                                              R_EIG_ratio=R_EIG_ratio, R_power=R_power,
                                              region_names=test_region_names, plot_freq=1, view=False)
    # ===================================================================================== Preparing data for testing
    # if PLOT:
    #     plt.plot(test_daily_cases.T)
    #     plt.savefig('./logs/' + folder + "/images/raw_test_data.png", bbox_inches='tight')
    #     plt.plot(test_daily_filtered.T)
    #     plt.savefig('./logs/' + folder + "/images/filtered_test_data.png", bbox_inches='tight')

    x_test_data, y_test_data, x_test_data_scalers = get_data(False, normalize=True,
                                                             data=test_daily_cases, dataf=test_daily_filtered,
                                                             population=test_population)
    x_dataf, y_dataf, x_data_scalersf = get_data(True, normalize=True,
                                                 data=test_daily_cases, dataf=test_daily_filtered,
                                                 population=test_population)
    # ================================================================================================= Initialize Model
    model, reduce_regions2batch = get_model(args.modeltype,
                                            input_days=WINDOW_LENGTH,
                                            output_days=PREDICT_STEPS,
                                            n_features=-1,
                                            n_regions=-1,
                                            show=True)

    fmodel_name = str(DATASETS) + "_" + model.name + "_" + TRAINING_DATA_TYPE + '_' + UNDERSAMPLING + '_' + str(
        model.input.shape[1]) + '_' + str(model.output.shape[1])
    if args.load_recent:
        try:
            model = tf.keras.models.load_model("models/" + fmodel_name + ".h5")
        except:
            pass

    print(fmodel_name)
    folder = time.strftime('%Y.%m.%d-%H.%M.%S', time.localtime()) + "_" + fmodel_name
    os.makedirs('./logs/' + folder + '/images')



    # ===================================================================================== Preparing data for training
    fil, raw, fs = load_multiple_data(DATASETS, args.path, look_back_window, window_slide, R_EIG_ratio, R_power,
                                      midpoint)
    if split_train_data:
        for i_region in range(len(fil)):
            to_keep = fil[i_region].shape[0] -(TEST_DAYS//window_slide)  # skip some days forcefully for testing
            # to_keep = (test_days_split_idx-look_back_window)//window_slide
            assert to_keep > 0
            if fil[i_region].shape[0] < to_keep:
                Warning(f"Region has {fil[i_region].shape[0]} to train, can't keep {to_keep} samples as train data.")
            else:
                print(
                    f"Total samples for {i_region} is {len(fil[i_region])}. Dropping last {fil[i_region].shape[0] - to_keep}")
                fil[i_region] = fil[i_region][:to_keep]
                raw[i_region] = raw[i_region][:to_keep]

    if TRAINING_DATA_TYPE == "Filtered":
        temp = load_samples(fil, fs, WINDOW_LENGTH, PREDICT_STEPS)
        x_train_list, y_train_list, x_test_list, y_test_list, x_val_list, y_val_list, fs_train, fs_test, fs_val = temp
    else:
        temp = load_samples(raw, fs, WINDOW_LENGTH, PREDICT_STEPS)
        x_train_list, y_train_list, x_test_list, y_test_list, x_val_list, y_val_list, fs_train, fs_test, fs_val = temp

    # ==================================================================================================== Undersampling
    print("================================================== Training data before undersampling")
    total_regions, total_samples = 0, 0
    for i in range(len(x_train_list)):  # (n_regions, samples*, WINDOW_LENGTH)
        total_regions += 1
        total_samples += x_train_list[i].shape[0]
    for i in range(len(x_test_list)):  # (n_regions, samples*, WINDOW_LENGTH)
        total_samples += x_test_list[i].shape[0]
    for i in range(len(x_val_list)):  # (n_regions, samples*, WINDOW_LENGTH)
        total_samples += x_val_list[i].shape[0]
    print(f"Total regions {total_regions} Total samples {total_samples}")

    if UNDERSAMPLING == "Reduce":
        # under-sampling parameters

        optimised = True
        clip = True

        # if optimised:
        #     if clip:
        #         clip_percentages = [0, 10]
        #     count_h, count_l, num_h, num_l = 2, 0.2, 100000, 500
        #     power_l, power_h, power_penalty = 0.2, 2, 1000
        # else:
        #     ratio = 0.3

        # if optimised:
        #     if clip:
        #         clip_percentages = [0, 10]
        #     count_h, count_l, num_h, num_l = 2, 0.2, 1000, 50
        #     # 10k, 500
        #     power_l, power_h, power_penalty = 0.2, 2, num_l
        # else:
        #     ratio = 0.3

        x_train_list, y_train_list, fs_train = undersample3(x_train_list, y_train_list, fs_train, window_slide, clip,
                                                            str(DATASETS), PLOT,
                                                            f'./logs/{folder}/images/under_{DATASETS}.png' if PLOT else None)

        print(f"Undersample percentage {x_train_list[0].shape[0] / total_samples * 100:.2f}%")
        # EPOCHS = min(250, int(EPOCHS * total_samples / x_train_list[0].shape[0]))
        print(f"New Epoch = {EPOCHS}")
        # here Xtrain have been reduced by regions

    # print("================================================= Training data after undersampling")
    # print("Train", x_train.shape, y_train.shape, x_train_feat.shape)
    # print("Val", x_val.shape, y_val.shape, x_val_feat.shape)
    # print("Test", x_test.shape, y_test.shape, x_test_feat.shape)
    if reduce_regions2batch:
        x_train_list, y_train_list, fs_train = reduce_regions_to_batch([x_train_list, y_train_list, fs_train])
        x_test_list, y_test_list, fs_test = reduce_regions_to_batch([x_test_list, y_test_list, fs_test])
        x_val_list, y_val_list, fs_val = reduce_regions_to_batch([x_val_list, y_val_list, fs_val])

        x_train, y_train, x_train_feat = expand_dims([x_train_list, y_train_list, fs_train], 3)
        x_test, y_test, x_test_feat = expand_dims([x_test_list, y_test_list, fs_test], 3)
        x_val, y_val, x_val_feat = expand_dims([x_val_list, y_val_list, fs_val], 3)
    else:
        raise NotImplementedError()

    # ============================================================================================== Creating Dataset
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    global freq, xcheck
    freq, xcheck = np.histogram(x_train.reshape((-1,)))  # np.concatenate(x_train, -1).mean(0))

    print("================================================== Training data after reducing shapes")
    print("Train", x_train.shape, y_train.shape, x_train_feat.shape)
    print("Val", x_val.shape, y_val.shape, x_val_feat.shape)
    print("Test", x_test.shape, y_test.shape, x_test_feat.shape)

    if PLOT:
        fig, axs = plt.subplots(2, 2)
        for _x_train in x_train_list:
            axs[0, 0].plot(_x_train)
        axs[0, 0].set_title("Original data")

        for i in range(x_train.shape[-1]):
            idx = np.random.randint(0, len(x_train), 100)
            axs[0, 1].plot(np.concatenate([x_train[idx, :, i], y_train[idx, :, i]], 1).T, linewidth=1)
        axs[0, 1].axvline(x_train.shape[1], color='r', linestyle='--')

        axs[1, 0].hist(x_train.reshape(-1), bins=100)
        axs[1, 0].set_title("Histogram of cases")

        axs[1, 1].hist(np.concatenate(x_train, -1).mean(0), bins=100)
        axs[1, 1].set_title("Histogram of mean of training samples")

        plt.savefig('./logs/' + folder + f"/images/Train_data.png", bbox_inches='tight')

    # =================================================================================================  Train

    train(model, train_data, x_train, y_train, x_test, y_test)

    # ================================================================================================= Few Evaluations
    model = tf.keras.models.load_model("models/" + fmodel_name + ".h5")
    if PLOT:
        test1(model, "Final")


def test1(model, epoch):
    global x_test_data_scalers
    n_regions = len(x_test_data_scalers.data_max_)

    def get_model_predictions(model, _x_data, _y_data, scalers):
        print(
            f"Predicting from model (in:{model.input.shape} out:{model.output.shape}). X={_x_data.shape} Y={_y_data.shape}")
        # CREATING TRAIN-TEST SETS FOR CASES
        _x_test, _y_test = split_into_pieces_inorder(_x_data.T, _y_data.T, WINDOW_LENGTH, PREDICT_STEPS,
                                                   WINDOW_LENGTH + PREDICT_STEPS,
                                                   reduce_last_dim=False)

        if model.input.shape[-1] == 1:
            _y_pred = np.zeros_like(_y_test)
            for i in range(len(test_region_names)):
                _y_pred[:, :, i] = model(_x_test[:, :, i:i + 1])[:, :, 0]
        else:
            _y_pred = model(_x_test).numpy()

        # # NOTE:
        # # max value may change with time. then we have to retrain the model!!!!!!
        # # we can have a predefined max value. 1 for major cities and 1 for smaller districts
        _x_test = undo_normalization(_x_test, scalers)
        _y_test = undo_normalization(_y_test, scalers)
        _y_pred = undo_normalization(_y_pred, scalers)

        return _x_test, _y_test, _y_pred

    x_data, y_data, _ = get_data(filtered=False, normalize=x_test_data_scalers, data=test_daily_cases, dataf=test_daily_filtered,
                                 population=test_population)
    x_test, y_test, y_pred = get_model_predictions(model, x_data, y_data, x_test_data_scalers)
    x_data, y_data, _ = get_data(filtered=True, normalize=x_test_data_scalers, data=test_daily_cases, dataf=test_daily_filtered,
                                 population=test_population)
    x_testf, y_testf, y_predf = get_model_predictions(model, x_data, y_data, x_test_data_scalers)

    Ys = np.stack([y_test, y_testf, y_pred, y_predf], 1)
    method_list = ['Observations Raw',
                   'Observations Filtered',
                   'Predicted using raw data',
                   'Predicted using Filtered data']
    styles = {
        'X': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
        'Xf': {'Preprocessing': 'Filtered', 'Data': 'Training', 'Size': 2},
        'Observations Raw': {'Preprocessing': 'Raw', 'Data': 'Training', 'Size': 2},
        'Observations Filtered': {'Preprocessing': 'Filtered', 'Data': 'Training', 'Size': 2},
        'Predicted using raw data': {'Preprocessing': 'Raw', 'Data': 'Predicted using raw data', 'Size': 4},
        'Predicted using Filtered data': {'Preprocessing': 'Filtered', 'Data': 'Predicted using Filtered data',
                                          'Size': 3},

    }
    # x_data, y_data = get_data(filtered=False, normalize=False)
    # region_mask = (np.mean(x_data,0) > 50).astype('int32')
    region_mask = (np.arange(n_regions) == 4).astype('int32')

    plt.figure(figsize=(20, 10))
    plot_prediction(x_test, x_testf, Ys, method_list, styles, test_region_names, region_mask)
    plt.title(str(epoch))
    # plt.savefig(f"./logs/{folder}/images/test1_{epoch}.eps")
    plt.savefig(f"./logs/{folder}/images/test1_{epoch}.png")


if __name__ == "__main__":
    main()
