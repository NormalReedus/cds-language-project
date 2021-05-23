import os
import math
import random

def load_data(data_path, test_split = 0.2, seed = 1, sample_num = None):
    # lines all start with a label that is 0, 1, 2, or 3
    lines_to_keep = ['0', '1', '2', '3'] 

    data = []

    with open(data_path, encoding='utf-8') as file:
        for line in file: 
            if line[0] in lines_to_keep:
                data_list = line.split('\t') # label and text is separated by a tab
                data_list[1] = data_list[1].replace('\n', '') # remove newlines
                data_list[1].strip() # remove other whitespace from comment
                data_list[0] = int(data_list[0]) # model wants numbers, not strings

                data.append(data_list)
    
    # collapse labels 1, 2, 3 into just 1
    data = reclassify_labels(data)

    # make sure we have the same number of 0 (neutral) as we have 1 (hate speech)
    data = equalize_labels(data, seed)

    # subsetting the data if we are doing a demo
    if sample_num:
        data = data[:sample_num]

    # train / test split
    split_point = math.floor(len(data) * (1 - test_split))
    train_data = data[:split_point]
    test_data = data[split_point:]

    return train_data, test_data

# use this if you only need toxic (0) and non-toxic (1) labels
def reclassify_labels(data):
    toxic_labels = [2, 3]

    data_copy = data.copy()

    # reclassify 2s and 3s to 1
    for i in range(len(data_copy)):
        if data_copy[i][0] in toxic_labels:
            data_copy[i][0] = 1
    
    return data_copy

# slices the neutral comment_lines (0) to the same number of data_points as hate speech (1)
def equalize_labels(data, seed = 1):
    random.seed(seed)

    # find all lines that are neutral
    neutral = [data_point for data_point in data if data_point[0] == 0]
    # find all lines that are hate speech
    hate = [data_point for data_point in data if data_point[0] == 1]

    # slice the neutrals to same length as hate speech
    random.shuffle(neutral)
    neutral = neutral[:len(hate)]

    # combine lists again and shuffle
    sliced_data = neutral + hate
    random.shuffle(sliced_data)

    return sliced_data