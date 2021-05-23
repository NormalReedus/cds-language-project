import os
import math
import random

def load_data(data_path, test_split = 0.2, seed = 1):
    # data points all start with a label that is 0, 1, 2, or 3
    lines_to_keep = ['0', '1', '2', '3'] 

    data = []

    with open(data_path, encoding='utf-8') as file:
        for line in file: 
            if line[0] in lines_to_keep:
                data_list = line.split('\t')
                data_list[1] = data_list[1].replace('\n', '') # remove newlines
                data_list[1].strip() # remove whitespace from comment
                data_list[0] = int(data_list[0])

                data.append(data_list)
    
    data = reclassify_labels(data)

    data = equalize_labels(data, seed)

    split_point = math.floor(len(data) * (1 - test_split))

    train_data = data[:split_point]
    test_data = data[split_point:]


    return train_data, test_data

# use this if you only need toxic (0) and non-toxic (1) labels
def reclassify_labels(data):
    toxic_labels = [2, 3]

    data_copy = data.copy()

    for i in range(len(data_copy)):
        if data_copy[i][0] in toxic_labels:
            data_copy[i][0] = 1
    
    return data_copy

# slices the neutral comment_lines (0) to the same number of data_points as hate speech (1)
def equalize_labels(data, seed = 1):
    random.seed(seed)

    neutral = [data_point for data_point in data if data_point[0] == 0]
    hate = [data_point for data_point in data if data_point[0] == 1]

    random.shuffle(neutral)
    neutral = neutral[:len(hate)]

    sliced_data = neutral + hate
    random.shuffle(sliced_data)

    return sliced_data