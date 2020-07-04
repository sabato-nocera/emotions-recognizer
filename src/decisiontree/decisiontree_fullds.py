# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

import sys
from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
from keras.utils import np_utils
from numpy import interp
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from er_utils import utils

logs_name = "../../logs/log_decisiontree_fullds_"

output_model_accuracy, output_model_loss, output_roc_curve_one, output_roc_curve_two, output_file_name = utils.files_name(
    logs_name)

output_file = open(output_file_name, "w")

sys.stdout = output_file

matplotlib.use('TkAgg')

n_classes = 4
n_header = 11
number_of_splits = 5
verbose_value = 2
shuffle_value = False
validation_split_value = 0.2
test_size_value = 0.2

max_depth_value = 2 ** 22

numpy.random.seed(7)


def kfold_cross_validation(X_train_kfold, X_test_kfold, y_train_kfold, y_test_kfold):
    inputs = np.concatenate((X_train_kfold, X_test_kfold), axis=0)
    targets = np.concatenate((y_train_kfold, y_test_kfold), axis=0)

    test_acc_per_fold = []
    train_acc_per_fold = []

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=number_of_splits, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1

    print('\n\n------------------------------------------------------------------------')
    print('K-fold Cross Validation')
    for train, test in kfold.split(inputs, targets):
        # create model
        tree_clf = DecisionTreeClassifier(max_depth=max_depth_value)

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        tree_clf.fit(inputs[train], targets[train])

        test_score_kfold = tree_clf.score(inputs[test], targets[test])
        train_score_kfold = tree_clf.score(inputs[train], targets[train])

        # Generate generalization metrics
        test_acc_per_fold.append(test_score_kfold * 100)
        train_acc_per_fold.append(train_score_kfold * 100)

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(test_acc_per_fold)):
        print('------------------------------------------------------------------------')
        print("Score for fold", i+1)
        print("Accuracy_Train: %.2f%%" % (train_acc_per_fold[i]))
        print("Accuracy_Test: %.2f%%" % (test_acc_per_fold[i]))
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print("Average_Accuracy_Train: %.2f%%" % (np.mean(train_acc_per_fold)))
    print("\t-> (+-", (np.std(train_acc_per_fold)), ")")
    print("Average_Accuracy_Test: %.2f%%" % (np.mean(test_acc_per_fold)))
    print("\t-> (+-", (np.std(test_acc_per_fold)), ")")
    print('------------------------------------------------------------------------')


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
dummy_x = scaler.fit_transform(X)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(dummy_x, dummy_y, test_size=test_size_value)

print("\nMax depth:", max_depth_value)

tree_clf = DecisionTreeClassifier(max_depth=max_depth_value)
print('\nStart computation...\n')

# returns the fitted estimator
tree_clf = tree_clf.fit(X_train, y_train)

print("Fitted estimator:\n", tree_clf)

print("Classes:")
for tmp in tree_clf.classes_:
    print("\t", tmp)
print("Feature importances:", tree_clf.feature_importances_)
print("The inferred value of max_features:", tree_clf.max_features)
print("The number of classes:", tree_clf.n_classes_)
print("The number of features:", tree_clf.n_features_)
print("The number of outputs:", tree_clf.n_outputs_)
print("The underlying Tree object:", tree_clf.tree_)

test_score = tree_clf.score(X_test, y_test)
train_score = tree_clf.score(X_train, y_train)

print("\nAccuracy Train: %.2f%%" % (train_score * 100))
print("Accuracy Test: %.2f%%" % (test_score * 100))

predictions = tree_clf.predict(X_test)

countPositive = 0
countNegative = 0

for i in range(len(X_test)):

    predicted_class = - 1

    for j in range(len(predictions[i])):
        if predictions[i][j] == 1:
            predicted_class = j

    if predicted_class == 0:
        if y_test[i][0] == 1:
            countPositive = countPositive + 1
        else:
            countNegative = countNegative + 1
    if predicted_class == 1:
        if y_test[i][1] == 1:
            countPositive = countPositive + 1
        else:
            countNegative = countNegative + 1
    if predicted_class == 2:
        if y_test[i][2] == 1:
            countPositive = countPositive + 1
        else:
            countNegative = countNegative + 1
    if predicted_class == 3:
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
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), predictions.ravel())
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

kfold_cross_validation(X_train, X_test, y_train, y_test)

output_file.close()
