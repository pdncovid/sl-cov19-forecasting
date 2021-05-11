import tensorflow as tf


def get_model(modeltype, input_days, output_days, n_features, n_regions):
    if modeltype == "Dense_WO_regions":
        reduce_regions2batch = True
        model = Dense_WO_regions(input_days, output_days, n_features)
    elif modeltype == "Dense_W_regions":
        reduce_regions2batch = False
        model = Dense_W_regions(input_days, output_days, n_features, n_regions)
    elif modeltype == "LSTM_Simple_WO_Regions":
        reduce_regions2batch = True
        model = LSTM_Simple_WO_Regions(input_days, output_days)
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
    model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True, rankdir='LR')
    return model, reduce_regions2batch


"""***Dense Models***"""


def Dense_WO_regions(seq_size, predict_steps, n_features):
    inp_seq = tf.keras.layers.Input(seq_size, name="input_sequence")
    inp_fea = tf.keras.layers.Input(n_features, name="input_features")

    x = inp_seq
    xf = inp_fea
    n = n_features
    while (n > 0):
        xf = tf.keras.layers.Dense(n, activation='relu')(xf)
        n = n // 2

    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(predict_steps, activation='relu')(x)

    if n_features > 0:
        x = x * xf
    model = tf.keras.models.Model([inp_seq, inp_fea], x, name="Dense_WO_regions")
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
    #     inp_fea = tf.keras.layers.Input(n_features, name="input_fea")

    x = inp_seq
    #     x = tf.keras.layers.LSTM(32, activation='relu', return_sequences=True)(x)
    x = tf.keras.layers.LSTM(output_seq_size)(x)
    # x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.Reshape((output_seq_size, 1))(x)
    model = tf.keras.models.Model(inp_seq, x, name="LSTM_Simple_WO_Regions")

    #     print("Input shape", X_train.shape[-2:])
    #     model = tf.keras.models.Sequential([
    #         tf.keras.layers.LSTM(128, input_shape=X_train.shape[-2:],dropout=dropout),
    #         tf.keras.layers.Dense(128),
    #         tf.keras.layers.Dense(output_seq_size),
    #         tf.keras.layers.Activation('sigmoid')
    #     ])

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
        out = xx if i == 0 else tf.keras.layers.concatenate([out, xx])

        xx = tf.reshape(xx, (-1, 1, n_regions))

        lstm_input = tf.keras.layers.concatenate([lstm_input[:, 1:, :], xx], axis=1)

    out = tf.reshape(out, (-1, output_seq_size, n_regions))
    #     out = tf.keras.layers.Activation('sigmoid')(out)
    model = tf.keras.models.Model(inp_seq, out, name="LSTM4EachDay_W_Regions")
    return model


def LSTM4EachDay_WO_Regions(input_seq_size, output_seq_size):
    inp_seq = tf.keras.layers.Input((input_seq_size, 1), name="input_seq")

    lstm_input = inp_seq
    for i in range(output_seq_size):
        xx = tf.keras.layers.LSTM(output_seq_size, activation='relu')(lstm_input)
        xx = xx[:, 0:1]
        out = xx if i == 0 else tf.keras.layers.concatenate([out, xx])

        xx = tf.reshape(xx, (-1, 1, 1))

        lstm_input = tf.keras.layers.concatenate([lstm_input[:, 1:, :], xx], axis=1)

    out = tf.reshape(out, (-1, output_seq_size, 1))
    #     out = tf.keras.layers.Activation('sigmoid')(out)
    model = tf.keras.models.Model(inp_seq, out, name="LSTM4EachDay_WO_Regions")
    return model