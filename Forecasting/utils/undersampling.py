import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Forecasting.utils.data_loader import reduce_regions_to_batch
from Forecasting.utils.functions import normalize_3d_xy_data


def min_max(data):
    for k in range(data.shape[0]):
        data[k, :] = (data[k, :] - np.amin(data[k, :])) / (np.amax(data[k, :]) - np.amin(data[k, :]))
    return data


def clip_dist(data_mean, data_list, clip_percentages):
    data = np.array(data_mean)
    b1 = data < np.quantile(data, clip_percentages[0] / 100)
    b2 = data >= np.quantile(data, (100 - clip_percentages[1]) / 100)
    idx = np.logical_or(b1, b2)
    data = np.delete(data, idx)
    list_out = [data]
    for list0 in data_list:
        list_out.append(np.delete(list0, idx, axis=0))
    return list_out, idx


def rejoin_dist(data_mean_new, data_list_new, data_mean_old, data_list_old, idx):

    data_mean = np.concatenate((data_mean_new, data_mean_old[idx]), axis=0)
    list_out = [data_mean]
    for n in range(len(data_list_new)):
        list0 = data_list_new[n]
        list1 = data_list_old[n]
        list_out.append(np.concatenate((list0, list1[idx, :]), axis=0))
    return list_out


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


def undersample3(x_data, y_data, x_feats, count_h, count_l, num_h, num_l, power_l, power_h, power_penalty,
                 clip, clip_percentages, country, PLOT,
                 savepath=None):
    global idx_clip
    print(f"Under-sampling! Expected data (regions, samples*, window).")
    total_regions, total_samples = 0, 0
    for i in range(len(x_data)):  # (n_regions, samples*, WINDOW_LENGTH)
        total_regions += 1
        total_samples += x_data[i].shape[0]

    x_data, y_data = normalize_3d_xy_data(x_data, y_data)
    x_init, y_init, f_init = reduce_regions_to_batch([x_data, y_data, x_feats])

    a = (count_l - count_h) / (num_h - num_l)
    b = count_h - (a * num_l)
    count_power_init = np.around(total_samples * a + b, 3)
    if count_power_init > power_h:
        count_power = power_h
    elif count_power_init < power_l:
        count_power = power_l
    else:
        count_power = count_power_init

    print('old samples = ' + str(total_samples) + '   count power (raw) = ' + str(
        count_power_init) + '   count power (fixed) = ' + str(count_power))

    x_mean_init = np.mean(x_init, axis=-1)

    if clip:
        [samples_x_mean, samples_x, samples_y, samples_f], idx_clip = clip_dist(x_mean_init, [x_init, y_init, f_init],
                                                                                clip_percentages)
    else:
        samples_x_mean, samples_x, samples_y, samples_f = x_mean_init, x_init, y_init, f_init

    # evaluating optimal number of segments for each district
    segment_array = np.rint(np.linspace(2, 20, 19)).astype(int)

    all_counts = []
    # evaluating the count score for each district
    for segments in segment_array:
        [count, _, _] = get_count(segments, samples_x_mean)
        all_counts.append(np.amin(count) * len(count))

    all_counts = np.array(all_counts)
    all_counts_norm = (all_counts - np.amin(all_counts)) / (np.amax(all_counts) - np.amin(all_counts))
    seg_score = np.linspace(1, len(segment_array), len(segment_array))
    count_score = np.multiply(all_counts_norm ** count_power, seg_score)
    count_score[all_counts < power_penalty] = 0.001 * count_score[all_counts < power_penalty]

    segments = segment_array[np.argmax(count_score)]

    if PLOT:
        plt.figure(figsize=(12, 8))
        plt.subplot(221)
        plt.stem(segment_array, all_counts)
        plt.title('all counts')
        plt.ylabel('total counts')
        plt.subplot(222)
        plt.plot(segment_array, count_score, linewidth=2)
        plt.ylabel('count score')
        plt.title('segments: ' + str(
            segment_array[np.argmax(count_score)]) + ' c_power: ' + str(count_power))

    # under-sampling using optimal number of segments
    [count_dist, _, idx_dist] = get_count(segments, samples_x_mean)
    n_per_seg = np.amin(count_dist)  # minimum samples from all segments

    idx_rand = np.zeros([segments, n_per_seg])  # random sampling from segment
    for k in range(segments):
        idx_temp = list(idx_dist[k])
        idx_rand[k, :] = random.sample(idx_temp, n_per_seg)

    idx_rand = np.reshape(idx_rand, [-1, ]).astype(int)

    x_train_opt = samples_x[idx_rand, :]
    y_train_opt = samples_y[idx_rand, :]
    x_train_fea = samples_f[idx_rand, :]
    x_mean_opt = samples_x_mean[idx_rand]

    if clip:
        [x_mean_opt, x_train_opt, y_train_opt, x_train_fea] = rejoin_dist(x_mean_opt,
                                                                          [x_train_opt, y_train_opt, x_train_fea],
                                                                          x_mean_init, [x_init, y_init, f_init],
                                                                          idx_clip)

    if PLOT:
        plt.subplot(223)
        plt.hist(x_mean_init, bins=50, alpha=0.5, label='original')
        plt.title('samples: ' + str(samples_x.shape[0]))
        plt.subplot(224)
        plt.hist(x_mean_opt, bins=50, alpha=0.5, label='original')
        plt.title(f'seg:{segments} n_per_seg: {n_per_seg} samples: {x_train_opt.shape[0]}')
        plt.suptitle('OPTIMISED UNDER-SAMPLING' + '\nregion: ' + str(country))
        plt.show()
        if savepath is not None:
            plt.savefig(savepath)
    print('new samples = ' + str(x_train_opt.shape[0]))
    # x_train_opt = np.expand_dims(x_train_opt, -1)
    # y_train_opt = np.expand_dims(y_train_opt, -1)
    # x_train_fea = np.expand_dims(x_train_fea, -1)

    return [x_train_opt], [y_train_opt], [x_train_fea]


def undersample_random(x_data, y_data, x_feats, ratio, country, PLOT, savepath=None):
    print(f"Random under-sampling! Expected data (regions, samples*, window).")
    total_regions, total_samples = 0, 0
    for i in range(len(x_data)):  # (n_regions, samples*, WINDOW_LENGTH)
        total_regions += 1
        total_samples += x_data[i].shape[0]

    x_data, y_data = normalize_3d_xy_data(x_data, y_data)
    samples_x, samples_y, samples_f = reduce_regions_to_batch([x_data, y_data, x_feats])

    samples_x_mean = np.mean(samples_x, axis=-1)

    samples_norm = (samples_x_mean - np.min(samples_x_mean)) / (np.max(samples_x_mean) - np.min(samples_x_mean))

    count_total = np.rint(ratio * len(samples_norm)).astype(int)
    print('initial samples = ' + str(len(samples_norm)) + '   final samples = ' + str(count_total))

    idx_rand = []
    for count in range(count_total):
        x_rand = np.around(np.max(samples_norm) * np.random.rand(), 5)
        err = np.abs(samples_norm - x_rand)
        idx_rand.append(err.argmin())

    x_train_opt = samples_x[idx_rand, :]
    y_train_opt = samples_y[idx_rand, :]
    x_train_fea = samples_f[idx_rand, :]

    # calculating unique values
    _, idx_unique = np.unique(np.mean(x_train_opt, axis=-1), return_index=True)
    x_unique = x_train_opt[idx_unique, :]
    x_repeated = np.delete(x_train_opt, idx_unique, axis=0)

    if PLOT:
        plt.figure(figsize=(12, 8))
        plt.subplot(221)
        plt.hist(samples_x_mean, bins=50, alpha=0.5, label='original')
        plt.title('initial samples: ' + str(len(samples_x_mean)))
        plt.subplot(222)
        plt.hist(np.mean(x_train_opt, axis=-1), bins=50, alpha=0.5, label='under-sampled')
        plt.title('ratio: ' + str(ratio) + '   final samples: ' + str(count_total))
        plt.subplot(223)
        plt.hist(np.mean(x_unique, axis=-1), bins=50, alpha=0.5, label='original')
        plt.title('unique samples: ' + str(x_unique.shape[0]))
        plt.subplot(224)
        plt.hist(np.mean(x_repeated, axis=-1), bins=50, alpha=0.5, label='original')
        plt.title('repeated samples: ' + str(x_repeated.shape[0]))
        plt.suptitle('RANDOM UNDER-SAMPLING' + '\nregion: ' + str(country))
        plt.show()
        if savepath is not None:
            plt.savefig(savepath)

    return [x_train_opt], [y_train_opt], [x_train_fea]

#
# def undersample(x_data, y_data, count_power, WINDOW_LENGTH, PREDICT_STEPS, region_names, PLOT, savepath=None):
#     print(f"Undersampling. Expectated data (regions, days). Got {x_data.shape}")
#     n_regions, days = x_data.shape
#
#     alldata_train = x_data
#
#     samples_all = np.zeros([n_regions, days - WINDOW_LENGTH - PREDICT_STEPS, WINDOW_LENGTH])
#     for i in range(n_regions):
#         for k in range(samples_all.shape[1]):
#             samples_all[i, k, :] = alldata_train[i, k:k + WINDOW_LENGTH]
#     samples_mean = np.mean(samples_all, axis=2)
#
#     # evaluating optimal number of segments for each district
#     segment_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     segment_dist = []
#     if PLOT:
#         plt.figure(figsize=(5 * 6, 5 * 4))
#         cols = min(n_regions, 5)
#
#     for i in range(n_regions):
#         all_counts = []
#         count_score = []
#         # evaluating the count score for each district
#         for n in range(len(segment_array)):
#             segments = segment_array[n]
#             [count, _, _] = get_count(segments, samples_mean[i, :])
#             all_counts.append(np.amin(count) * len(count))
#             count_score.append((all_counts[n] ** count_power) * (n + 1))
#
#         segment_dist.append(segment_array[np.argmax(count_score)])
#
#         if PLOT:
#             plt.subplot(np.ceil(samples_all.shape[0] / cols).astype(int), cols, i + 1)
#
#             plt.plot(segment_array, all_counts / np.amax(all_counts), linewidth=2)
#             plt.plot(segment_array, count_score / np.amax(count_score), linewidth=2)
#             plt.legend(['normalised total counts', 'segment score'])
#             plt.title('dist: ' + region_names[i] + '  segments: ' + str(
#                 segment_array[np.argmax(count_score)]) + '  samples: ' + str(all_counts[np.argmax(count_score)]))
#
#     segment_dist = np.array(segment_dist).astype(int)
#
#     if PLOT:
#         plt.show()
#
#     # print('segments per district= ', segment_dist)
#
#     x_train_opt, y_train_opt = [], []
#     plt.figure(figsize=(5 * 6, 5 * 4))
#     # undersampling using optimal number of segments
#     for i in range(n_regions):
#         segments = segment_dist[i]
#         [count_dist, bounds_dist, idx_dist] = get_count(segments, samples_mean[i, :])
#         n_per_seg = np.amin(count_dist)  # minimum samples from all segments
#
#         idx_rand = np.zeros([segments, n_per_seg])  # random sampling from segment
#         for k in range(segments):
#             idx_temp = list(idx_dist[k])
#             idx_rand[k, :] = random.sample(idx_temp, n_per_seg)
#         idx_rand = np.reshape(idx_rand, [-1, ]).astype(int)
#
#         if PLOT:
#             plt.subplot(np.ceil(samples_all.shape[0] / cols).astype(int), cols, i + 1)
#             bins = np.linspace(0, np.max(samples_mean[i, :]), 20)
#
#             plt.hist(samples_mean[i, :], bins=bins, histtype='step')
#             plt.hist(samples_mean[i, idx_rand], bins=bins, histtype='step')
#
#         for j, idx in enumerate(idx_rand):
#             x_train_opt.append(x_data[i, idx:idx + WINDOW_LENGTH])
#             y_train_opt.append(y_data[i, idx + WINDOW_LENGTH:idx + WINDOW_LENGTH + PREDICT_STEPS])
#             if PLOT:
#                 rect = patches.Rectangle((idx, j / len(idx_rand)), WINDOW_LENGTH, 1 / len(idx_rand), linewidth=1,
#                                          edgecolor=None, facecolor=(1., 0., 0., 0.5))
#                 ax.add_patch(rect)
#                 rect = patches.Rectangle((idx + WINDOW_LENGTH, j / len(idx_rand)), PREDICT_STEPS, 1 / len(idx_rand),
#                                          linewidth=1, edgecolor=None, facecolor=(0., 1., 0., 0.5))
#                 ax.add_patch(rect)
#
#     if PLOT and savepath is not None:
#         plt.savefig(savepath)
#
#     x_train_opt = np.array(x_train_opt)
#     y_train_opt = np.array(y_train_opt)
#
#     x_train_opt = np.expand_dims(x_train_opt, -1)
#     y_train_opt = np.expand_dims(y_train_opt, -1)
#     return x_train_opt, y_train_opt
#
#
# def undersample2(x_train, y_train, x_feats, count_power, region_names, PLOT, savepath=None):
#     print(f"Undersampling. Expectated data (samples, window, regions). Got {x_train.shape}")
#
#     x_train = x_train.transpose([2, 0, 1])
#     y_train = y_train.transpose([2, 0, 1])
#
#     n_regions, samples, WINDOW_LENGTH = x_train.shape
#     samples_all = x_train
#     samples_mean = np.mean(samples_all, axis=2)
#
#     # evaluating optimal number of segments for each district
#     segment_array = [2, 3, 4, 5, 6, 7, 8, 9, 10]
#     segment_dist = []
#     if PLOT:
#         plt.figure(figsize=(5 * 6, 5 * 4))
#
#     for i in range(n_regions):
#         all_counts = []
#         count_score = []
#         # evaluating the count score for each district
#         for n in range(len(segment_array)):
#             segments = segment_array[n]
#             [count, bounds, idx] = get_count(segments, samples_mean[i, :])
#             all_counts.append(np.amin(count) * len(count))
#             count_score.append((all_counts[n] ** count_power) * (n + 1))
#
#         segment_dist.append(segment_array[np.argmax(count_score)])
#
#         if PLOT:
#             plt.subplot(np.ceil(samples_all.shape[0] / 5).astype(int), 5, i + 1)
#             plt.plot(segment_array, all_counts / np.amax(all_counts), linewidth=2)
#             plt.plot(segment_array, count_score / np.amax(count_score), linewidth=2)
#             plt.legend(['normalised total counts', 'segment score'])
#             plt.title('dist: ' + region_names[i] + '  segments: ' + str(
#                 segment_array[np.argmax(count_score)]) + '  samples: ' + str(all_counts[np.argmax(count_score)]))
#
#     segment_dist = np.array(segment_dist).astype(int)
#     if savepath is not None:
#         plt.savefig(savepath + 'logic.png')
#
#     x_train_opt, y_train_opt = [], []
#     feats = []
#     plt.figure(figsize=(5 * 6, 5 * 4))
#     # under-sampling using optimal number of segments
#     for i in range(n_regions):
#         segments = segment_dist[i]
#         [count_dist, bounds_dist, idx_dist] = get_count(segments, samples_mean[i, :])
#         n_per_seg = np.amin(count_dist)  # minimum samples from all segments
#
#         idx_rand = np.zeros([segments, n_per_seg])  # random sampling from segment
#         for k in range(segments):
#             idx_temp = list(idx_dist[k])
#             idx_rand[k, :] = random.sample(idx_temp, n_per_seg)
#         idx_rand = np.reshape(idx_rand, -1).astype(int)
#
#         if PLOT:
#             plt.subplot(np.ceil(samples_all.shape[0] / 5).astype(int), 5, i + 1)
#             bins = np.linspace(0, np.max(samples_mean[i, :]), 20)
#
#             plt.hist(samples_mean[i, :], bins=bins, histtype='step')
#             plt.hist(samples_mean[i, idx_rand], bins=bins, histtype='step')
#         for j, idx in enumerate(idx_rand):
#             x_train_opt.append(x_train[i, idx, :])
#             y_train_opt.append(y_train[i, idx, :])
#             feats.append(x_feats[0, :, i])
#     if savepath is not None:
#         plt.savefig(savepath)
#
#     x_train_opt = np.array(x_train_opt)
#     y_train_opt = np.array(y_train_opt)
#     feats = np.array(feats)
#     x_train_opt = np.expand_dims(x_train_opt, -1)
#     y_train_opt = np.expand_dims(y_train_opt, -1)
#     feats = np.expand_dims(feats, -1)
#     print(f"Undersmapled shape {x_train_opt.shape} {y_train_opt.shape} {feats.shape}")
#     return x_train_opt, y_train_opt, feats
