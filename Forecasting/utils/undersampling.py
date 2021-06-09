import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def min_max(data):
    for k in range(data.shape[0]):
        data[k, :] = (data[k, :] - np.amin(data[k, :])) / (np.amax(data[k, :]) - np.amin(data[k, :]))
    return data


def get_count(segments, data):
    bounds = []
    count = []
    idx = []
    for i in range(segments):
        data = (data - np.amin(data))
        bounds.append(np.round((i + 1) * np.amax(data) / segments, 3))
        if i == 0:
            ineq = data <= bounds[i]
        elif i == (segments - 1):
            ineq = data > bounds[i - 1]
        else:
            ineq = (data > bounds[i - 1]) * (data <= bounds[i])
        count.append(np.sum(ineq))
        idx.append(np.reshape(np.array(np.where(ineq)), [-1, ]))
    count = np.array(count).astype(int)
    bounds = np.array(bounds).astype(np.float64)
    return count, bounds, idx


def undersample(x_data, y_data, count_power, WINDOW_LENGTH, PREDICT_STEPS, region_names, PLOT, savepath=None):
    print(f"Undersampling. Expectated data (regions, days). Got {x_data.shape}")
    n_regions, days = x_data.shape

    alldata_train = x_data

    samples_all = np.zeros([n_regions, days - WINDOW_LENGTH - PREDICT_STEPS, WINDOW_LENGTH])
    for i in range(n_regions):
        for k in range(samples_all.shape[1]):
            samples_all[i, k, :] = alldata_train[i, k:k + WINDOW_LENGTH]
    samples_mean = np.mean(samples_all, axis=2)

    # evaluating optimal number of segments for each district
    segment_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    segment_dist = []
    if PLOT:
        plt.figure(figsize=(5 * 6, 5 * 4))
        cols = min(n_regions, 5)

    for i in range(n_regions):
        all_counts = []
        count_score = []
        # evaluating the count score for each district
        for n in range(len(segment_array)):
            segments = segment_array[n]
            [count, bounds, idx] = get_count(segments, samples_mean[i, :])
            all_counts.append(np.amin(count) * len(count))
            count_score.append((all_counts[n] ** count_power) * (n + 1))

        segment_dist.append(segment_array[np.argmax(count_score)])

        if PLOT:

            plt.subplot(np.ceil(samples_all.shape[0] / cols).astype(int), cols, i + 1)

            plt.plot(segment_array, all_counts / np.amax(all_counts), linewidth=2)
            plt.plot(segment_array, count_score / np.amax(count_score), linewidth=2)
            plt.legend(['normalised total counts', 'segment score'])
            plt.title('dist: ' + region_names[i] + '  segments: ' + str(
                segment_array[np.argmax(count_score)]) + '  samples: ' + str(all_counts[np.argmax(count_score)]))

    segment_dist = np.array(segment_dist).astype(int)

    if PLOT:
        plt.show()

    # print('segments per district= ', segment_dist)

    x_train_opt, y_train_opt = [], []
    plt.figure(figsize=(5 * 6, 5 * 4))
    # undersampling using optimal number of segments
    for i in range(n_regions):
        segments = segment_dist[i]
        [count_dist, bounds_dist, idx_dist] = get_count(segments, samples_mean[i, :])
        n_per_seg = np.amin(count_dist)  # minimum samples from all segments

        idx_rand = np.zeros([segments, n_per_seg])  # random sampling from segment
        for k in range(segments):
            idx_temp = list(idx_dist[k])
            idx_rand[k, :] = random.sample(idx_temp, n_per_seg)
        idx_rand = np.reshape(idx_rand, [-1, ]).astype(int)

        if PLOT:

            plt.subplot(np.ceil(samples_all.shape[0] / cols).astype(int), cols, i + 1)
            bins = np.linspace(0, np.max(samples_mean[i, :]), 20)

            plt.hist(samples_mean[i, :], bins=bins, histtype='step')
            plt.hist(samples_mean[i, idx_rand], bins=bins, histtype='step')

        for j, idx in enumerate(idx_rand):
            x_train_opt.append(x_data[i, idx:idx + WINDOW_LENGTH])
            y_train_opt.append(y_data[i, idx + WINDOW_LENGTH:idx + WINDOW_LENGTH + PREDICT_STEPS])
            if PLOT:
                rect = patches.Rectangle((idx, j / len(idx_rand)), WINDOW_LENGTH, 1 / len(idx_rand), linewidth=1,
                                         edgecolor=None, facecolor=(1., 0., 0., 0.5))
                ax.add_patch(rect)
                rect = patches.Rectangle((idx + WINDOW_LENGTH, j / len(idx_rand)), PREDICT_STEPS, 1 / len(idx_rand),
                                         linewidth=1, edgecolor=None, facecolor=(0., 1., 0., 0.5))
                ax.add_patch(rect)

    if PLOT and savepath is not None:
        plt.savefig(savepath)

    x_train_opt = np.array(x_train_opt)
    y_train_opt = np.array(y_train_opt)

    x_train_opt = np.expand_dims(x_train_opt, -1)
    y_train_opt = np.expand_dims(y_train_opt, -1)
    return x_train_opt, y_train_opt


def undersample2(x_train, y_train, x_feats, count_power, region_names, PLOT, savepath=None):
    print(f"Undersampling. Expectated data (samples, window, regions). Got {x_train.shape}")
    x_train = x_train.transpose([2, 0, 1])
    y_train = y_train.transpose([2, 0, 1])

    n_regions, samples, WINDOW_LENGTH = x_train.shape
    samples_all = x_train
    samples_mean = np.mean(samples_all, axis=2)

    # evaluating optimal number of segments for each district
    segment_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    segment_dist = []
    if PLOT:
        plt.figure(figsize=(5 * 6, 5 * 4))

    for i in range(n_regions):
        all_counts = []
        count_score = []
        # evaluating the count score for each district
        for n in range(len(segment_array)):
            segments = segment_array[n]
            [count, bounds, idx] = get_count(segments, samples_mean[i, :])
            all_counts.append(np.amin(count) * len(count))
            count_score.append((all_counts[n] ** count_power) * (n + 1))

        segment_dist.append(segment_array[np.argmax(count_score)])

        if PLOT:
            plt.subplot(np.ceil(samples_all.shape[0] / 5).astype(int), 5, i + 1)
            plt.plot(segment_array, all_counts / np.amax(all_counts), linewidth=2)
            plt.plot(segment_array, count_score / np.amax(count_score), linewidth=2)
            plt.legend(['normalised total counts', 'segment score'])
            plt.title('dist: ' + region_names[i] + '  segments: ' + str(
                segment_array[np.argmax(count_score)]) + '  samples: ' + str(all_counts[np.argmax(count_score)]))

    segment_dist = np.array(segment_dist).astype(int)
    if savepath is not None:
        plt.savefig(savepath + 'logic.png')

    # print('segments per district= ', segment_dist)

    x_train_opt, y_train_opt = [], []
    feats = []
    plt.figure(figsize=(5 * 6, 5 * 4))
    # undersampling using optimal number of segments
    for i in range(n_regions):
        segments = segment_dist[i]
        [count_dist, bounds_dist, idx_dist] = get_count(segments, samples_mean[i, :])
        n_per_seg = np.amin(count_dist)  # minimum samples from all segments

        idx_rand = np.zeros([segments, n_per_seg])  # random sampling from segment
        for k in range(segments):
            idx_temp = list(idx_dist[k])
            idx_rand[k, :] = random.sample(idx_temp, n_per_seg)
        idx_rand = np.reshape(idx_rand, -1).astype(int)

        if PLOT:
            plt.subplot(np.ceil(samples_all.shape[0] / 5).astype(int), 5, i + 1)
            bins = np.linspace(0, np.max(samples_mean[i, :]), 20)

            plt.hist(samples_mean[i, :], bins=bins, histtype='step')
            plt.hist(samples_mean[i, idx_rand], bins=bins, histtype='step')
        for j, idx in enumerate(idx_rand):
            x_train_opt.append(x_train[i, idx, :])
            y_train_opt.append(y_train[i, idx, :])
            feats.append(x_feats[0,:,i])
    if savepath is not None:
        plt.savefig(savepath)

    x_train_opt = np.array(x_train_opt)
    y_train_opt = np.array(y_train_opt)
    feats = np.array(feats)
    x_train_opt = np.expand_dims(x_train_opt, -1)
    y_train_opt = np.expand_dims(y_train_opt, -1)
    feats = np.expand_dims(feats, -1)
    print(f"Undersmapled shape {x_train_opt.shape} {y_train_opt.shape} {feats.shape}")
    return x_train_opt, y_train_opt, feats


def undersample3(x_data, y_data, count_power, WINDOW_LENGTH, PREDICT_STEPS, region_names, PLOT, savepath=None):
    print(f"Undersampling. Expected data (regions, days). Got {x_data.shape}")

    n_regions, days = x_data.shape

    alldata_train = min_max(x_data)

    # print(type(alldata_train))

    samples_all = np.zeros([n_regions, days - WINDOW_LENGTH - PREDICT_STEPS, WINDOW_LENGTH + PREDICT_STEPS])

    for i in range(n_regions):
        for k in range(samples_all.shape[1]):
            samples_all[i, k, :] = alldata_train[i, k:k + WINDOW_LENGTH + PREDICT_STEPS]

    samples_all = np.reshape(samples_all, (samples_all.shape[0] * samples_all.shape[1], samples_all.shape[2]))
    # print('samples_all.shape=', samples_all.shape)
    samples_mean = np.mean(samples_all, axis=-1)
    # print('samples_mean.shape=', samples_mean.shape)
    # if PLOT:
    #     plt.figure()
    #     plt.hist(samples_mean)
    #     plt.show()
    # evaluating optimal number of segments for each district
    segment_array = np.rint(np.linspace(2, 20, 10)).astype(int)

    all_counts = []
    count_score = []
    # evaluating the count score for each district
    for n in range(len(segment_array)):
        segments = segment_array[n]
        [count, _, _] = get_count(segments, samples_mean)
        # print('\nsegments:', segments)
        count_dist = np.amin(count) * len(count)
        # print('count_dist ' + str(count_dist))
        all_counts.append(count_dist)
        # print('all counts ' + str(all_counts))

    all_counts = np.array(all_counts)
    all_counts_norm = (all_counts - np.amin(all_counts)) / (np.amax(all_counts) - np.amin(all_counts))
    seg_score = np.linspace(1, len(segment_array), len(segment_array))
    count_score = np.multiply(all_counts_norm ** count_power, seg_score)
    count_score[all_counts < 1000] = 0.2*count_score[all_counts < 1000]
    # print('seg_score', seg_score)
    # for n in range(len(segment_array)):
    #     count_score.append((np.array(all_counts_norm[n]) ** count_power) * (n + 1))

    segment_dist = segment_array[np.argmax(count_score)]
    # print('optimal segments= ', segment_dist)

    if PLOT:
        plt.figure(figsize=(12, 8))
        plt.subplot(221)
        plt.stem(segment_array, all_counts)
        plt.title('all counts')
        plt.ylabel('total counts')
        # plt.subplot(1, 3, 2)
        # plt.plot(segment_array, np.array(all_counts) ** count_power, linewidth=2)
        # plt.title('count_power: ' + str(count_power))
        # plt.ylabel('total counts^count_power')
        plt.subplot(222)
        plt.plot(segment_array, count_score, linewidth=2)
        plt.ylabel('count score')
        plt.title('segments: ' + str(
            segment_array[np.argmax(count_score)]) + ' c_power: '+str(count_power))
        # plt.show()

    x_train_opt, y_train_opt = [], []

    # undersampling using optimal number of segments

    segments = segment_dist
    [count_dist, _, idx_dist] = get_count(segment_dist, samples_mean)
    n_per_seg = np.amin(count_dist)  # minimum samples from all segments

    idx_rand = np.zeros([segments, n_per_seg])  # random sampling from segment
    for k in range(segments):
        idx_temp = list(idx_dist[k])
        idx_rand[k, :] = random.sample(idx_temp, n_per_seg)
    # print('idx_rand.shape=', idx_rand.shape)
    idx_rand = np.reshape(idx_rand, [-1, ]).astype(int)
    # print('idx_rand.shape=', idx_rand.shape)

    # for j, idx in enumerate(idx_rand):
    #     x_train_opt.append(samples_all[idx:idx + WINDOW_LENGTH])
    #     y_train_opt.append(y_data[idx + WINDOW_LENGTH:idx + WINDOW_LENGTH + PREDICT_STEPS])
    x_train_opt = samples_all[idx_rand, 0:WINDOW_LENGTH]
    y_train_opt = samples_all[idx_rand, WINDOW_LENGTH:WINDOW_LENGTH + PREDICT_STEPS]

    if PLOT:
        # plt.figure(figsize=(12, 4))
        plt.subplot(223)
        plt.hist(np.mean(samples_all, axis=-1), bins=50), plt.title('samples: ' + str(samples_all.shape[0]))
        plt.subplot(224)
        plt.hist(np.mean(x_train_opt, axis=-1), bins=50), plt.title(
            'seg:' + str(segments) + ' samples: ' + str(x_train_opt.shape[0]))
        plt.show()

    if PLOT and savepath is not None:
        plt.savefig(savepath)

    # x_train_opt = np.array(x_train_opt)
    # y_train_opt = np.array(y_train_opt)

    x_train_opt = np.expand_dims(x_train_opt, -1)
    y_train_opt = np.expand_dims(y_train_opt, -1)

    # print('after expand dims', x_train_opt.shape, y_train_opt.shape)

    return x_train_opt, y_train_opt
