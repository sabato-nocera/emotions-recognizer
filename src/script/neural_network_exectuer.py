from itertools import cycle

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas
from keras.utils import np_utils
from numpy import interp
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from er_utils import utils

h5_name = "../models/mlp_secondaugmentedb_categoricalcrossentropy_adam_model_2020-06-14_22.17.34.h5"
json_name = "../models/mlp_secondaugmentedb_categoricalcrossentropy_adam_model_2020-06-14_22.17.34.json"

# Loading Keras model
neural_network_json_file = open(json_name)
neural_network_json = neural_network_json_file.read()
neural_network_json_file.close()
neural_network = keras.models.model_from_json(neural_network_json)

# Loading weights for Keras model
neural_network.load_weights(h5_name)

neural_network.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

dataset_name = "../datasets/full_dataset_without_humidity.csv"
dataframe = pandas.read_csv(dataset_name, header=0, sep=";", skiprows=0)
print("Dataset used:", dataset_name, "\n")
print(dataframe.head())

print("\nObjservations: {}".format(len(dataframe)))
dataset = dataframe.values

n_header = 10
n_classes = 4

X = dataset[:, 0:n_header]
X = X.astype('float32')

Y = dataset[:, n_header]
scaler = MinMaxScaler(feature_range=(-1, 1))
dummy_x = scaler.fit_transform(X)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# using the full dataset as test set
X_test = dummy_x
y_test = dummy_y

print(len(X_test))
print((len(y_test)))

y_score = neural_network.predict(X_test)
test_score = neural_network.evaluate(X_test, y_test, verbose=2)
print("Accuracy Test: %.2f%%" % (test_score[1] * 100))
print("Loss Test: %.2f" % (test_score[0]))

predictions = neural_network.predict_classes(X_test)

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
plt.show()
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
plt.show()
plt.clf()
