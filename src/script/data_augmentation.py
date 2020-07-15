import numpy
import pandas
import random

n_header = 10


def tuples_crossover(first_tuple, second_tuple):
    if first_tuple[10] != second_tuple[10]:
        print("err")
        return []

    new_tuples_crossover = []

    # splitting the features by five
    first = numpy.array([first_tuple[0:6], second_tuple[0:6]])
    second = numpy.array([first_tuple[6:10], second_tuple[6:10]])

    # creating every new possibilities
    tmp_tuple = numpy.concatenate((first[0], second[1], first_tuple[10:]))
    new_tuples_crossover.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], first_tuple[10:]])
    new_tuples_crossover.append(tmp_tuple)

    new_tuples_crossover = numpy.array(new_tuples_crossover)

    return new_tuples_crossover


# loading dataset
dataset_name = "../../datasets/other/to_augment.csv"
dataframe = pandas.read_csv(dataset_name, header=0, sep=";", skiprows=0)
print("\nDataset used:", dataset_name, "\n")
print(dataframe.head())

print("\nObjservations: {}".format(len(dataframe)))
dataset = dataframe.values

# CROSSING-OVER
#
# we consider tuples linked to the same feedback from the same person
number_of_users = int(len(dataset) / 80)

grouped_by_user = []

i = 0
for k in range(number_of_users):
    grouped_by_user.append(dataset[i:i + 80])
    i += 80

new_tuples = numpy.array([])

feature_temperature = 0
feature_heartbeat = 2

# 5% standard deviation
# range_temperature = 0.803827
# range_heartbeat = 0.643367

# 5% variance
range_temperature = 13
range_heartbeat = 8

for i in range(len(dataset)):
    new_tuple = numpy.array(dataset[i])

    new_tuple[feature_temperature] += random.choice([i for i in range(-range_temperature,+range_temperature) if i not in [-2,-1,0,1,2,2]])
    new_tuple[feature_heartbeat] += random.choice([i for i in range(-range_heartbeat,+range_heartbeat) if i not in [-1,0,1]])
    new_tuples = numpy.concatenate((new_tuples, new_tuple), axis=0)

    new_tuple[feature_temperature] += random.choice([i for i in range(-range_temperature,+range_temperature) if i not in [-2,-1,0,1,2,2]])
    new_tuple[feature_heartbeat] += random.choice([i for i in range(-range_heartbeat,+range_heartbeat) if i not in [-1,0,1]])
    new_tuples = numpy.concatenate((new_tuples, new_tuple), axis=0)

# for tmp_list in grouped_by_user:
#     last_list = []
#     for tmp_array in tmp_list:
#         if len(last_list) == 0:
#             last_list = tmp_array
#             continue
#         if tmp_array[10] == last_list[10]:
#             tmp_new_tuples = tuples_crossover(tmp_array, last_list)
#             new_tuple_one = numpy.array(tmp_new_tuples[0])
#             new_tuple_two = numpy.array(tmp_new_tuples[1])
#
#             new_tuple_one[feature_temperature] += random.uniform(-range_temperature, +range_temperature)
#             new_tuple_one[feature_heartbeat] += random.uniform(-range_heartbeat, +range_heartbeat)
#
#             new_tuple_two[feature_temperature] += random.uniform(-range_temperature, +range_temperature)
#             new_tuple_two[feature_heartbeat] += random.uniform(-range_heartbeat, +range_heartbeat)
#
#             new_tuples = numpy.concatenate((new_tuples, new_tuple_one), axis=0)
#             new_tuples = numpy.concatenate((new_tuples, new_tuple_two), axis=0)
#         last_list = tmp_array

new_tuples = new_tuples.reshape((int(len(new_tuples) / 11), 11))

print("New tuples created:", len(new_tuples))

for tmp in new_tuples:
    data_to_append = {}
    for i in range(len(dataframe.columns)):
        data_to_append[dataframe.columns[i]] = tmp[i]
    dataframe = dataframe.append(data_to_append, ignore_index=True)

print(len(dataframe))

print(dataframe)

dataframe.to_csv("../../datasets/other/train_dataset_for_augmentation.csv")
