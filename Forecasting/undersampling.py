import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def undersample(x_data, y_data, WINDOW_LENGTH, PREDICT_STEPS, region_names, PLOT):
    print(f"Undersampling. Expectated data (regions, days). Got {x_data.shape}")
    n_regions = x_data.shape[0]
    dataset_norm = np.zeros_like(x_data)
    for i in range(n_regions):
        dataset_norm[i, :] = x_data[i, :] / np.amax(x_data[i, :])

    alldata_train = dataset_norm

    samples_all = np.zeros(
        [alldata_train.shape[0], alldata_train.shape[1] - WINDOW_LENGTH - PREDICT_STEPS, WINDOW_LENGTH])
    samples_mean = np.zeros([alldata_train.shape[0], alldata_train.shape[1] - WINDOW_LENGTH - PREDICT_STEPS])

    # evaluating optimal number of segments for each district
    segment_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    segment_dist = []
    if PLOT:
        plt.figure(figsize=(5 * 6, 5 * 4))
    for i in range(samples_all.shape[0]):
        for k in range(samples_all.shape[1]):
            samples_all[i, k, :] = alldata_train[i, k:k + WINDOW_LENGTH]
            samples_mean[i, k] = np.mean(samples_all[i, k, :])
        all_counts = []
        count_score = []
        # evaluating the count score for each district
        for n in range(len(segment_array)):
            segments = segment_array[n]
            [count, bounds, idx] = get_count(segments, samples_mean[i, :])
            all_counts.append(np.amin(count) * len(count))
            count_score.append((all_counts[n] ** 1) * (n + 1))

        segment_dist.append(segment_array[np.argmax(count_score)])

        if PLOT:
            plt.subplot(5, 5, i + 1)
            plt.plot(segment_array, all_counts / np.amax(all_counts), linewidth=2)
            plt.plot(segment_array, count_score / np.amax(count_score), linewidth=2)
            plt.legend(['normalised total counts', 'segment score'])
            plt.title('dist: ' + region_names[i] + '  segments: ' + str(
                segment_array[np.argmax(count_score)]) + '  samples: ' + str(all_counts[np.argmax(count_score)]))

    segment_dist = np.array(segment_dist).astype(int)

    if PLOT:
        plt.show()

    print('segments per district= ', segment_dist)

    idx_rand_all = []
    for i in range(samples_all.shape[0]):
        data = samples_mean[i, :]
        segments = segment_dist[i]
        [count_dist, bounds_dist, idx_dist] = get_count(segments, data)
        n_per_seg = np.amin(count_dist)
        data_new = []
        idx_rand = np.zeros([segments, n_per_seg])
        for k in range(segments):
            idx_temp = list(idx_dist[k])
            idx_rand[k, :] = random.sample(idx_temp, n_per_seg)
        idx_rand = np.reshape(idx_rand, [-1, ])
        idx_rand_all.append(idx_rand)
    print(len(idx_rand_all))

    x_train_opt, y_train_opt = [], []
    # undersampling using optimal number of segments
    for i in range(samples_all.shape[0]):
        data = samples_mean[i, :]
        segments = segment_dist[i]
        [count_dist, bounds_dist, idx_dist] = get_count(segments, data)
        n_per_seg = np.amin(count_dist)
        data_new = []
        idx_rand = np.zeros([segments, n_per_seg])
        for k in range(segments):
            idx_temp = list(idx_dist[k])
            #         print(idx_temp)
            idx_rand[k, :] = random.sample(idx_temp, n_per_seg)
        idx_rand = np.reshape(idx_rand, [-1, ])
        print(region_names[i], idx_rand)

        if PLOT:
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(x_data[i, :])
        for j, idx in enumerate(idx_rand):
            idx = int(idx)

            if PLOT:
                rect = patches.Rectangle((idx, j / len(idx_rand)), WINDOW_LENGTH, 1 / len(idx_rand), linewidth=1,
                                         edgecolor=None, facecolor=(1., 0., 0., 0.5))
                ax.add_patch(rect)
                rect = patches.Rectangle((idx + WINDOW_LENGTH, j / len(idx_rand)), PREDICT_STEPS, 1 / len(idx_rand),
                                         linewidth=1, edgecolor=None, facecolor=(0., 1., 0., 0.5))
                ax.add_patch(rect)

            x_train_opt.append(x_data[i, idx:idx + WINDOW_LENGTH])
            y_train_opt.append(y_data[i, idx + WINDOW_LENGTH:idx + WINDOW_LENGTH + PREDICT_STEPS])
        if PLOT:
            plt.show()

    x_train_opt = np.array(x_train_opt)
    y_train_opt = np.array(y_train_opt)

    x_train_opt = np.expand_dims(x_train_opt, -1)
    y_train_opt = np.expand_dims(y_train_opt, -1)
    return x_train_opt, y_train_opt