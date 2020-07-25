from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def files_name(name):
    now = datetime.now()
    output_file_name = name + str(now)
    i = output_file_name.rindex(".")
    output_file_name = output_file_name[0:i]
    output_file_name = output_file_name.replace(":", ".")
    output_file_name = output_file_name.replace(" ", "_")

    output_model_accuracy = output_file_name + "_model_accuracy.png"
    output_model_loss = output_file_name + "_model_loss.png"
    output_roc_curve_one = output_file_name + "_roc_curve_one.png"
    output_roc_curve_two = output_file_name + "_roc_curve_two.png"
    output_file_name = output_file_name + ".txt"

    return output_model_accuracy, output_model_loss, output_roc_curve_one, output_roc_curve_two, output_file_name


def models_name(name):
    now = datetime.now()

    json_file_name = name + str(now)
    i = json_file_name.rindex(".")
    json_file_name = json_file_name[0:i]
    json_file_name = json_file_name.replace(":", ".")
    json_file_name = json_file_name.replace(" ", "_")
    json_file_name = json_file_name + ".json"

    hdf5_file_name = name + str(now)
    i = hdf5_file_name.rindex(".")
    hdf5_file_name = hdf5_file_name[0:i]
    hdf5_file_name = hdf5_file_name.replace(":", ".")
    hdf5_file_name = hdf5_file_name.replace(" ", "_")
    hdf5_file_name = hdf5_file_name + ".h5"

    return json_file_name, hdf5_file_name


def calculate(x, i, j):
    temp = []
    for i in range(i, j):
        temp.append(x[i])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dummy_x = scaler.fit_transform(temp)

    return dummy_x


def normalize(x):
    init = 0
    end = 80

    g = calculate(x, init, end)

    init = init + 80
    end = end + 80

    d = (len(x) - 80) / 80

    s = int(d)

    for k in range(s):
        r = calculate(x, init, end)
        g = np.concatenate((g, r), axis=0)
        init = init + 80
        end = end + 80

    return g


def smaller_normalization(x):
    init = 0
    end = 40

    g = calculate(x, init, end)

    init = init + 40
    end = end + 40

    d = (len(x) - 40) / 40

    s = int(d)

    for k in range(s):
        r = calculate(x, init, end)
        g = np.concatenate((g, r), axis=0)
        init = init + 40
        end = end + 40

    return g


def greatest_normalization(x):
    init = 0
    end = 214

    g = calculate(x, init, end)

    init = init + 214
    end = end + 214

    d = (len(x) - 214) / 214

    s = int(d)

    for k in range(s):
        r = calculate(x, init, end)
        g = np.concatenate((g, r), axis=0)
        init = init + 214
        end = end + 214

    return g


def greater_normalization(x):
    init = 0
    end = 107

    g = calculate(x, init, end)

    init = init + 107
    end = end + 107

    d = (len(x) - 107) / 107

    s = int(d)

    for k in range(s):
        r = calculate(x, init, end)
        g = np.concatenate((g, r), axis=0)
        init = init + 107
        end = end + 107

    return g
