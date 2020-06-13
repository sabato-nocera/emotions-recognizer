import pandas
import numpy

n_header = 10


def tuple_crossover_one(first_tuple, second_tuple):
    if first_tuple[10] != second_tuple[10]:
        print("err")
        return []

    new_tuples = []

    # splitting the features by five
    first = numpy.array([first_tuple[0:6], second_tuple[0:6]])
    second = numpy.array([first_tuple[6:10], second_tuple[6:10]])

    # creating every new possibilities
    tmp_tuple = numpy.concatenate((first[0], second[1], first_tuple[10:]))
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)

    new_tuples = numpy.array(new_tuples)

    return new_tuples


# creates maybe too many tuples...
def tuple_crossover_two(first_tuple, second_tuple):
    if first_tuple[10] != second_tuple[10]:
        print("err")
        return []

    new_tuples = []

    # splitting the features by two
    first = numpy.array([first_tuple[0:2], second_tuple[0:2]])
    second = numpy.array([first_tuple[2:4], second_tuple[2:4]])
    third = numpy.array([first_tuple[4:6], second_tuple[4:6]])
    fourth = numpy.array([first_tuple[6:8], second_tuple[6:8]])
    fifth = numpy.array([first_tuple[8:10], second_tuple[8:10]])

    # creating every new possibilities
    tmp_tuple = numpy.concatenate((first[0], second[0], third[0], fourth[0], fifth[1], first_tuple[10:]))
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[0], third[0], fourth[1], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[0], third[0], fourth[1], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[0], third[1], fourth[0], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[0], third[1], fourth[0], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[0], third[1], fourth[1], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[0], third[1], fourth[1], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[1], third[0], fourth[0], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[1], third[0], fourth[0], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[1], third[0], fourth[1], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[1], third[0], fourth[1], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[1], third[1], fourth[0], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[1], third[1], fourth[0], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[1], third[1], fourth[1], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[0], second[1], third[1], fourth[1], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], third[0], fourth[0], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], third[0], fourth[0], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], third[0], fourth[1], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], third[0], fourth[1], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], third[1], fourth[0], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], third[1], fourth[0], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], third[1], fourth[1], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[0], third[1], fourth[1], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[1], third[0], fourth[0], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[1], third[0], fourth[0], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[1], third[0], fourth[1], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[1], third[0], fourth[1], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[1], third[1], fourth[0], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[1], third[1], fourth[0], fifth[1], first_tuple[10:]])
    new_tuples.append(tmp_tuple)
    tmp_tuple = numpy.concatenate([first[1], second[1], third[1], fourth[1], fifth[0], first_tuple[10:]])
    new_tuples.append(tmp_tuple)

    new_tuples = numpy.array(new_tuples)

    return new_tuples


# loading dataset
dataset_name = "../../datasets/full_dataset_without_humidity.csv"
dataframe = pandas.read_csv(dataset_name, header=0, sep=";", skiprows=0)
print("\nDataset used:", dataset_name, "\n")
print(dataframe.head())

print("\nObjservations: {}".format(len(dataframe)))
dataset = dataframe.values

# we consider tuples linked to the same feedback from the same person
number_of_users = int(len(dataset) / 80)

grouped_by_user = []

i = 0
for k in range(number_of_users):
    grouped_by_user.append(dataset[i:i + 80])
    i += 80

new_tuples = numpy.array([])

for tmp_list in grouped_by_user:
    last_list = []
    for tmp_array in tmp_list:
        if len(last_list) == 0:
            last_list = tmp_array
            continue
        if tmp_array[10] == last_list[10]:
            tmp_new_tuples = tuple_crossover_one(tmp_array, last_list)

            new_tuples = numpy.concatenate((new_tuples, tmp_new_tuples[0]), axis=0)
            new_tuples = numpy.concatenate((new_tuples, tmp_new_tuples[1]), axis=0)
        last_list = tmp_array

new_tuples = new_tuples.reshape((int(len(new_tuples) / 11), 11))

print("New tuples created:", len(new_tuples))

for tmp in new_tuples:
    data_to_append = {}
    for i in range(len(dataframe.columns)):
        data_to_append[dataframe.columns[i]] = tmp[i]
    dataframe = dataframe.append(data_to_append, ignore_index=True)

print(len(dataframe))

print(dataframe)

dataframe.to_csv("../../datasets/second_augmented_dataset.csv")
