import tensorflow as tf
import numpy as np
import math
import pydot


def get_model(modeltype, input_days, output_days, n_features, n_regions, show=False):
    if modeltype == "Dense_WO_regions":
        reduce_regions2batch = True
        model = Dense_WO_regions(input_days, output_days, n_features)
    elif modeltype == "Dense_W_regions":
        reduce_regions2batch = False
        model = Dense_W_regions(input_days, output_days, n_features, n_regions)
    elif modeltype == "LSTM_Simple_WO_Regions":
        reduce_regions2batch = True
        model = LSTM_Simple_WO_Regions(input_days, output_days)
    elif modeltype == "LSTM_Simple_WO_Regions_v2":
        reduce_regions2batch = True
        model = LSTM_Simple_WO_Regions_v2(input_days, output_days)
    elif modeltype == "LSTM_Simple_W_Regions":
        reduce_regions2batch = False
        model = LSTM_Simple_W_Regions(input_days, output_days, n_regions)
    elif modeltype == "LSTM4EachDay_W_Regions":
        reduce_regions2batch = False
        model = LSTM4EachDay_W_Regions(input_days, output_days, n_regions)
    elif modeltype == "LSTM4EachDay_WO_Regions":
        reduce_regions2batch = True
        model = LSTM4EachDay_WO_Regions(input_days, output_days)
    else:
        raise TypeError("Model type not defined")

    if show:
        model.summary()
        # tf.keras.utils.plot_model(model, show_shapes=True, rankdir='LR')

    return model, reduce_regions2batch


"""***Dense Models***"""


def Dense_WO_regions(seq_size, predict_steps, n_features):
    inp_seq = tf.keras.layers.Input((seq_size, 1), name="input_sequence")
    x = tf.keras.layers.Reshape((seq_size,))(inp_seq)
    # inp_fea = tf.keras.layers.Input(n_features, name="input_features")

    # xf = inp_fea
    # n = n_features
    # while (n > 0):
    #     xf = tf.keras.layers.Dense(n, activation='relu')(xf)
    #     n = n // 2

    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(predict_steps, activation='relu')(x)

    # if n_features > 0:
    #     x = x * xf
    # model = tf.keras.models.Model([inp_seq, inp_fea], x, name="Dense_WO_regions")
    x = tf.keras.layers.Reshape((predict_steps, 1))(x)
    model = tf.keras.models.Model(inp_seq, x, name="Dense_WO_regions")
    return model


def Dense_W_regions(input_seq_size, output_seq_size, n_features, n_regions):
    inp_seq = tf.keras.layers.Input((input_seq_size, n_regions), name="input_sequence")
    inp_fea = tf.keras.layers.Input((n_features, n_regions), name="input_features")

    x = tf.keras.layers.Reshape((input_seq_size * n_regions,))(inp_seq)
    x = tf.keras.layers.Dense(output_seq_size * n_regions, activation='sigmoid')(x)
    x = tf.keras.layers.Reshape((output_seq_size, n_regions))(x)

    model = tf.keras.models.Model([inp_seq, inp_fea], x, name="Dense_W_regions")
    return model


"""***LSTM MODEL***"""


def LSTM_Simple_WO_Regions(input_seq_size, output_seq_size):
    inp_seq = tf.keras.layers.Input((input_seq_size, 1), name="input_seq")
    x = inp_seq
    cells = int(np.ceil((input_seq_size - output_seq_size) * 0.4 + output_seq_size))
    x = tf.keras.layers.LSTM(cells)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(output_seq_size)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Reshape((output_seq_size, 1))(x)
    model = tf.keras.models.Model(inp_seq, x, name="LSTM_Simple_WO_Regions")

    return model


def LSTM_Simple_WO_Regions_v2(input_seq_size, output_seq_size):
    inp_seq = tf.keras.layers.Input((input_seq_size, 1), name="input_seq")
    max_cells = 15
    x = inp_seq
    if input_seq_size > max_cells:
        cells = max_cells
    else:
        cells = int(np.ceil((input_seq_size - output_seq_size) * 0.8 + output_seq_size))
    x = tf.keras.layers.LSTM(cells)(x)
    x = tf.keras.layers.Activation('softmax')(x)
    x = tf.keras.layers.Dense(output_seq_size)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Reshape((output_seq_size, 1))(x)
    model = tf.keras.models.Model(inp_seq, x, name="LSTM_Simple_WO_Regions_v2")

    return model


def LSTM_Simple_W_Regions(input_seq_size, output_seq_size, n_regions):
    inp_seq = tf.keras.layers.Input((input_seq_size, n_regions), name="input_seq")
    x = tf.keras.layers.LSTM(output_seq_size * n_regions, activation='sigmoid')(inp_seq)
    x = tf.keras.layers.Reshape((output_seq_size, n_regions))(x)

    #     x = tf.keras.layers.Activation('sigmoid')(x)
    model = tf.keras.models.Model(inp_seq, x, name="LSTM_Simple_W_Regions")

    return model


def LSTM4EachDay_W_Regions(input_seq_size, output_seq_size, n_regions):
    inp_seq = tf.keras.layers.Input((input_seq_size, n_regions), name="input_seq")

    lstm_input = inp_seq
    for i in range(output_seq_size):
        xx = tf.keras.layers.LSTM(n_regions, activation='relu')(lstm_input)
        out = xx if i == 0 else tf.keras.layers.concatenate([out, xx], 1)

        xx = tf.reshape(xx, (-1, 1, n_regions))

        lstm_input = tf.keras.layers.concatenate([lstm_input[:, 1:, :], xx], axis=1)

    out = tf.reshape(out, (-1, output_seq_size, n_regions))
    #     out = tf.keras.layers.Activation('sigmoid')(out)
    model = tf.keras.models.Model(inp_seq, out, name="LSTM4EachDay_W_Regions")
    return model


def LSTM4EachDay_WO_Regions(input_seq_size, output_seq_size):
    k = 5
    assert output_seq_size % k == 0

    inp_seq = tf.keras.layers.Input((input_seq_size, 1), name="input_seq")
    lstm_input = inp_seq
    for i in range(0, output_seq_size, k):
        pre_cells = output_seq_size - i
        xx = tf.keras.layers.LSTM(pre_cells)(lstm_input)
        xx = xx[:, 0:k]
        xx = tf.reshape(xx, (-1, k, 1))

        out = xx if i == 0 else tf.keras.layers.concatenate([out, xx], axis=1)

        lstm_input = tf.keras.layers.concatenate([lstm_input[:, k:, :], xx], axis=1)

    x = tf.keras.layers.Activation('relu')(out)
    x = tf.keras.layers.Reshape((output_seq_size, 1))(x)
    # out = tf.keras.layers.Activation('sigmoid')(out)
    model = tf.keras.models.Model(inp_seq, x, name="LSTM4EachDay_WO_Regions")
    return model
