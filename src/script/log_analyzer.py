import ast
import glob
import sys
from datetime import datetime

# return the content of interest of a file
def read_file_content(file):
    file_content = "{"
    with open(file, 'rt') as fd:
        offset = 16
        file_name = file[offset:]
        file_content += "'file_name':'" + file_name + "'"

        for line in fd:

            if "Accuracy Test" in line:
                file_content += ","
                file_content += "'accuracy_test':" + line[-7:].rstrip("\n") + ""
                file_content = file_content[:-1]

            if "Accuracy Train" in line:
                file_content += ","
                file_content += "'accuracy_train':" + line[-7:].rstrip("\n") + ""
                file_content = file_content[:-1]

            if "Loss Test" in line:
                file_content += ","
                file_content += "'loss_test':" + line[-5:].rstrip("\n") + ""

            if "Average_Accuracy_Train" in line:
                file_content += ","
                file_content += "'average_accuracy_train':" + line[-7:].rstrip("\n") + ""
                file_content = file_content[:-1]

            if "Average_Accuracy_Test" in line:
                file_content += ","
                file_content += "'average_accuracy_test':" + line[-7:].rstrip("\n") + ""
                file_content = file_content[:-1]

            if "Average_Loss_Train" in line:
                file_content += ","
                file_content += "'average_loss_train':" + line[-5:].rstrip("\n") + ""

            if "Average_Loss_Test" in line:
                file_content += ","
                file_content += "'average_loss_test':" + line[-5:].rstrip("\n") + ""

    file_content += "},"
    return file_content. \
        strip()


# return the content of the files of a directory
def merge_per_folder(folder_path):
    folder_path += "" if folder_path[-1] == "/" else "/"

    txt_files = glob.glob(folder_path + "*.txt")

    output_strings = map(read_file_content, sorted(txt_files))
    output_content = "".join(output_strings)

    return output_content


def print_max_accuracy_test(list_of_dicts):
    max_accuracy = {"file_name": "", "value": 0}

    for tmp in list_of_dicts:
        if "accuracy_test" in tmp:
            if tmp["accuracy_test"] > max_accuracy["value"]:
                max_accuracy["value"] = tmp["accuracy_test"]
                max_accuracy["file_name"] = tmp["file_name"]

    print("\nMax accuracy test: ", max_accuracy["file_name"], " with ", max_accuracy["value"], "%", "(", tmp["date"],
          ")")


def print_min_loss(list_of_dicts):
    min_loss = {"file_name": "", "value": 999}

    for tmp in list_of_dicts:
        if "loss_test" in tmp:
            if tmp["loss_test"] < min_loss["value"]:
                min_loss["value"] = tmp["loss_test"]
                min_loss["file_name"] = tmp["file_name"]

    print("\nMin loss test: ", min_loss["file_name"], " with ", min_loss["value"], "(", tmp["date"], ")")


def print_sorted_by_name(list_of_dicts):
    sorted_by_name = sorted(list_of_dicts, key=lambda k: k['file_name'])

    print("\nSorted by name:\n")
    rank = 1
    for tmp in sorted_by_name:
        print(rank, ")", tmp["file_name"], " with accuracy test:", tmp["accuracy_test"], "% and accuracy train:",
              tmp["accuracy_train"], "%", "(", tmp["date"], ")")
        rank += 1


def print_sorted_by_accuracy_test(list_of_dicts):
    sorted_by_accuracy = sorted(list_of_dicts, key=lambda k: k['accuracy_test'])

    print("\nSorted by accuracy test:\n")
    rank = 1
    for tmp in sorted_by_accuracy:
        print(rank, ")", tmp["file_name"], " with ", tmp["accuracy_test"], "%", "(", tmp["date"], ")")
        rank += 1

def print_sorted_by_accuracy_test_without_augmented(list_of_dicts):
    sorted_by_accuracy = sorted(list_of_dicts, key=lambda k: k['accuracy_test'])

    print("\nSorted by accuracy test:\n")
    rank = 1
    for tmp in sorted_by_accuracy:
        if "augmented" in tmp["file_name"]:
            continue
        print(rank, ")", tmp["file_name"], " with ", tmp["accuracy_test"], "%", "(", tmp["date"], ")")
        rank += 1


def print_sorted_by_accuracy_train(list_of_dicts):
    sorted_by_accuracy = sorted(list_of_dicts, key=lambda k: k['accuracy_train'])

    print("\nSorted by accuracy train:\n")
    rank = 1
    for tmp in sorted_by_accuracy:
        print(rank, ")", tmp["file_name"], " with ", tmp["accuracy_train"], "%", "(", tmp["date"], ")")
        rank += 1


def print_sorted_by_loss(list_of_dicts):
    # not all dictionaries have the "loss_test" key
    sorted_by_loss = []
    for tmp in list_of_dicts:
        if "loss_test" in tmp:
            sorted_by_loss.append(tmp)
    sorted_by_loss = sorted(sorted_by_loss, key=lambda k: k['loss_test'])

    print("\nSorted by loss:\n")
    rank = 1
    for tmp in sorted_by_loss:
        print(rank, ")", tmp["file_name"], " with ", tmp["loss_test"], "(", tmp["date"], ")")
        rank += 1


def print_average_accuracy_test(list_of_dicts):
    accuracies_sum = 0
    i = 0
    for tmp in list_of_dicts:
        if "accuracy_test" in tmp:
            accuracies_sum += tmp["accuracy_test"]
            i += 1

    print("\nAverage accuracy test of logs:", round(accuracies_sum / i, 2), "%")


def print_above_percentage(list_of_dicts):
    above_80 = 0
    above_85 = 0
    above_86 = 0
    above_87 = 0
    above_88 = 0
    above_89 = 0
    above_90 = 0

    for tmp in list_of_dicts:
        if "accuracy_test" in tmp:
            if tmp["accuracy_test"] >= 80:
                above_80 += 1
            if tmp["accuracy_test"] >= 85:
                above_85 += 1
            if tmp["accuracy_test"] >= 86:
                above_86 += 1
            if tmp["accuracy_test"] >= 87:
                above_87 += 1
            if tmp["accuracy_test"] >= 88:
                above_88 += 1
            if tmp["accuracy_test"] >= 89:
                above_89 += 1
            if tmp["accuracy_test"] >= 90:
                above_90 += 1

    print("\nModels with test accuracy >= 80% :", above_80)
    print("\nModels with test accuracy >= 85% :", above_85)
    print("\nModels with test accuracy >= 86% :", above_86)
    print("\nModels with test accuracy >= 87% :", above_87)
    print("\nModels with test accuracy >= 88% :", above_88)
    print("\nModels with test accuracy >= 89% :", above_89)
    print("\nModels with test accuracy >= 90% :", above_90)


def print_by_loss_function(list_of_dicts, loss_function_name):
    print("\nWhich neural networks use the loss function '", loss_function_name, "' (sorted by accuracy test) ?\n")

    sorted_by_accuracy = []
    for tmp in list_of_dicts:
        if loss_function_name in tmp["file_name"]:
            sorted_by_accuracy.append(tmp)

    sorted_by_accuracy = sorted(sorted_by_accuracy, key=lambda k: k['accuracy_test'])
    rank = 1
    for tmp in sorted_by_accuracy:
        print(rank, ")", tmp["file_name"], " with ", tmp["accuracy_test"], "%", "(", tmp["date"], ")")
        rank += 1


def print_sorted_by_name_and_accuracy(list_of_dicts):
    sorted_by_name_and_accuracy = sorted(list_of_dicts, key=lambda k: (k['file_name'], k['accuracy_test']))

    print("\nSorted by name and accuracy:\n")
    rank = 1
    for tmp in sorted_by_name_and_accuracy:
        print(rank, ")", tmp["file_name"], " with accuracy test:", tmp["accuracy_test"], "%", "(", tmp["date"], ")")
        rank += 1


def print_ratio_loss_accuracy(list_of_dicts):
    print("\nLoss / Accuracy Test:\n")
    loss_on_accuracy = []
    for tmp in list_of_dicts:
        if "accuracy_test" in tmp and "loss_test" in tmp:
            tmp["ratio"] = tmp["loss_test"] / tmp["accuracy_test"]
            loss_on_accuracy.append(tmp)

    # not all dictionaries have the "loss_test" key
    sorted_by_loss = []
    for tmp in list_of_dicts:
        if "loss_test" in tmp:
            sorted_by_loss.append(tmp)
    sorted_by_loss = sorted(sorted_by_loss, key=lambda k: k['loss_test'])

    rank = 1
    sorted_by_ratio = sorted(sorted_by_loss, key=lambda k: k['ratio'])
    for tmp in sorted_by_ratio:
        print(rank, ")", tmp["file_name"], "\n\t-> Ratio:", tmp["ratio"], "-> Accuracy Test:",
              tmp["accuracy_test"], "% -> Loss Test:", tmp["loss_test"], "-> Accuracy Train:", tmp["accuracy_train"],
              "%", "(", tmp["date"], ")")
        rank += 1


def print_ratio_by_max_accuracy_for_each_type_of_neural_network(list_of_dicts):
    sorted_by_name_and_accuracy = sorted(list_of_dicts, key=lambda k: (k['file_name'], k['accuracy_test']))

    types_of_neural_network = []
    last_one = sorted_by_name_and_accuracy[0]
    for tmp in sorted_by_name_and_accuracy:
        if tmp["file_name"] != last_one["file_name"]:
            types_of_neural_network.append(last_one)
        last_one = tmp

    types_of_neural_network.append(last_one)

    for tmp in types_of_neural_network:
        if "accuracy_test" in tmp and "loss_test" in tmp:
            tmp["ratio"] = tmp["loss_test"] / tmp["accuracy_test"]
        else:
            tmp["loss_test"] = 100
            tmp["ratio"] = 100

    rank = 1
    sorted_by_ratio = sorted(types_of_neural_network, key=lambda k: k['ratio'])

    print("\nLoss / Accuracy Test (by max accuracy for each type of neural network):\n")
    rank = 1
    for tmp in sorted_by_ratio:
        print(rank, ")", tmp["file_name"], "\n\t-> Ratio:", tmp["ratio"], "-> Accuracy Test:",
              tmp["accuracy_test"], "% -> Loss Test:", tmp["loss_test"], "-> Accuracy Train:", tmp["accuracy_train"],
              "%", "(", tmp["date"], ")")
        rank += 1


def max_accuracy_for_each_type_of_neural_network(list_of_dicts):
    sorted_by_name_and_accuracy = sorted(list_of_dicts, key=lambda k: (k['file_name'], k['accuracy_test']))

    types_of_neural_network = []
    last_one = sorted_by_name_and_accuracy[0]
    for tmp in sorted_by_name_and_accuracy:
        if tmp["file_name"] != last_one["file_name"]:
            types_of_neural_network.append(last_one)
        last_one = tmp

    types_of_neural_network.append(last_one)

    sorted_accuracy = sorted(types_of_neural_network, key=lambda k: (k['accuracy_test']))

    print("\nMax accuracy for each type of neural network:\n")
    rank = 1
    for tmp in sorted_accuracy:
        print(rank, ")", tmp["file_name"], " with accuracy test:", tmp["accuracy_test"], "%", "(", tmp["date"], ")")
        print("\t", tmp)
        rank += 1

# now = datetime.now()
# output_file_name = "../../logs/log_analyzer_" + str(now)
# i = output_file_name.rindex(".")
# output_file_name = output_file_name[0:i]
# output_file_name = output_file_name.replace(":", ".")
# output_file_name = output_file_name.replace(" ", "_")
# output_file_name = output_file_name + ".txt"
# output_file = open(output_file_name, "w")
#
# sys.stdout = output_file

files_content = merge_per_folder("../../logs/past")

files_content = files_content[:-1]
files_content = "[" + files_content + "]"

file_info = ast.literal_eval(files_content)

for tmp in file_info:
    i = tmp["file_name"].index("2")
    tmp["date"] = tmp["file_name"][i:len(tmp["file_name"]) - 4].strip()
    tmp["file_name"] = tmp["file_name"][0:i - 1].strip()
print("\nNumber of logs:", len(file_info))

# print_by_loss_function(file_info, "categoricalcrossentropy")
# print_sorted_by_accuracy_train(file_info)
# print_sorted_by_loss(file_info)
# print_average_accuracy_test(file_info)
# print_above_percentage(file_info)
#
# print_sorted_by_name(file_info)
# print_max_accuracy_test(file_info)
# print_min_loss(file_info)
# print_ratio_loss_accuracy(file_info)
#
# print_sorted_by_name_and_accuracy(file_info)
#
# max_accuracy_for_each_type_of_neural_network(file_info)
#
# print_ratio_by_max_accuracy_for_each_type_of_neural_network(file_info)

# print_sorted_by_accuracy_test(file_info)

print_sorted_by_accuracy_test_without_augmented(file_info)

# output_file.close()
