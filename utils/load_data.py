import os
import math
import random

def load_data(data_path, test_split = 0.2, seed = 1):
    # data points all start with a label that is 0, 1, 2, or 3
    lines_to_keep = ['0', '1', '2', '3'] 

    data = []

    with open(data_path) as file:
        for line in file: 
            if line[0] in lines_to_keep:
                data_list = line.split('\t')
                data_list[1] = data_list[1].replace('\n', '') # remove newlines
                data_list[1].strip() # remove whitespace from comment
                data_list[0] = int(data_list[0])

                data.append(data_list)


    random.seed(seed)
    random.shuffle(data)

    split_point = math.floor(len(data) * (1 - test_split))

    train_data = data[:split_point]
    test_data = data[split_point:]

    train_data_reclassed, test_data_reclassed = reclassify_labels(train_data, test_data)

    return train_data_reclassed, test_data_reclassed

# use this if you only need toxic (0) and non-toxic (1) labels
def reclassify_labels(train_data, test_data):
    toxic_labels = [2, 3]

    train_copy = train_data.copy()
    test_copy = test_data.copy()

    for i in range(len(train_copy)):
        if train_copy[i][0] in toxic_labels:
            train_copy[i][0] = 1

    for i in range(len(test_copy)):
        if test_copy[i][0] in toxic_labels:
            test_copy[i][0] = 1
    
    return train_copy, test_copy
