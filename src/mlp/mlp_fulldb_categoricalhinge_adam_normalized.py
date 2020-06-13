import matplotlib
import numpy
import numpy as np
import pandas
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from numpy import interp
from itertools import cycle
from datetime import datetime
import sys


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


now = datetime.now()
output_file_name = "../../logs/log_mlp_fulldb_categoricalhinge_adam_normalized_" + str(now)
i = output_file_name.rindex(".")
output_file_name = output_file_name[0:i]
output_file_name = output_file_name.replace(":", ".")
output_file_name = output_file_name.replace(" ", "_")
output_model_accuracy = output_file_name + "_model_accuracy.png"
output_model_loss = output_file_name + "_model_loss.png"
output_roc_curve_one = output_file_name + "_roc_curve_one.png"
output_roc_curve_two = output_file_name + "_roc_curve_two.png"
output_file_name = output_file_name + ".txt"
output_file = open(output_file_name, "w")

sys.stdout = output_file

matplotlib.use('TkAgg')

n_classes = 4
n_header = 11

numpy.random.seed(7)


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(n_header, activation='tanh', input_dim=n_header))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    # compile model
    model.compile(loss='categorical_hinge', optimizer='adam', metrics=['accuracy'])

    print("\nLayers:\n")
    layers = model.layers
    for x in layers:
        print(x.get_config(), "\n")

    print("Compile: loss='categorical_hinge', optimizer='adam', metrics=['accuracy']")

    return model


dataset_name = "../../datasets/full_dataset.csv"
dataframe = pandas.read_csv(dataset_name, header=0, sep=";", skiprows=0)
print("Dataset used:", dataset_name, "\n")
print(dataframe.head())

print("\nObjservations: {}".format(len(dataframe)))
dataset = dataframe.values

X = dataset[:, 0:n_header]
X = X.astype('float32')

Y = dataset[:, n_header]
scaler = MinMaxScaler(feature_range=(-1, 1))
dummy_x = normalize(X)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(dummy_x, dummy_y, test_size=0.2)

keras_model = baseline_model()

print('\nStart computation...\n')

history = keras_model.fit(X_train, y_train, epochs=500, batch_size=80, verbose=2, shuffle=False, validation_split=0.20)
print("\nFit: epochs=500, batch_size=80, verbose=2, shuffle=False, validation_split=0.20\n")
print(keras_model.summary())

y_score = keras_model.predict(X_test)
test_score = keras_model.evaluate(X_test, y_test, verbose=2)
train_score = keras_model.evaluate(X_train, y_train, verbose=2)
print("\nAccuracy Train: %.2f%%" % (train_score[1] * 100))
print("Accuracy Test: %.2f%%" % (test_score[1] * 100))
print("Loss Train: %.2f" % (train_score[0]))
print("Loss Test: %.2f" % (test_score[0]))

predictions = keras_model.predict_classes(X_test)

countPositive = 0
countNegative = 0

for i in range(len(X_test)):
    if predictions[i] == 0:
        if y_test[i][0] == 1:
            countPositive = countPositive + 1
        else:
            countNegative = countNegative + 1
    if predictions[i] == 1:
        if y_test[i][1] == 1:
            countPositive = countPositive + 1
        else:
            countNegative = countNegative + 1
    if predictions[i] == 2:
        if y_test[i][2] == 1:
            countPositive = countPositive + 1
        else:
            countNegative = countNegative + 1
    if predictions[i] == 3:
        if y_test[i][3] == 1:
            countPositive = countPositive + 1
        else:
            countNegative = countNegative + 1

print("Numero dati esaminati: %d" % len(X_test))
print("True Positive %d" % countPositive)
print("False Positive %d" % countNegative)

# save the model
model_json = keras_model.to_json()
json_file_name = "../../models/mlp_fulldb_categoricalhinge_adam_normalized_model_" + str(now)
i = json_file_name.rindex(".")
json_file_name = json_file_name[0:i]
json_file_name = json_file_name.replace(":", ".")
json_file_name = json_file_name.replace(" ", "_")
json_file_name = json_file_name + ".json"
with open(json_file_name, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
hdf5_file_name = "../../models/mlp_fulldb_categoricalhinge_adam_normalized_model_" + str(now)
i = hdf5_file_name.rindex(".")
hdf5_file_name = hdf5_file_name[0:i]
hdf5_file_name = hdf5_file_name.replace(":", ".")
hdf5_file_name = hdf5_file_name.replace(" ", "_")
hdf5_file_name = hdf5_file_name + ".h5"
keras_model.save_weights(hdf5_file_name)
# print("\nSaved model to disk\n")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig(output_model_accuracy)
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig(output_model_loss)
plt.clf()

lw = 2

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
# plt.show()
plt.savefig(output_roc_curve_one)
plt.clf()

# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
# plt.show()
plt.savefig(output_roc_curve_two)
plt.clf()

output_file.close()