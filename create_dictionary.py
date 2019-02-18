import os
import re
from collections import Counter


def split_line(line):
    return re.split(r'[.?\-",;\'\r\n ]+', line)


# Make a dictionary
def make_dictionary(train_dir):
    articles = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for article in articles:
        with open(article) as m:
            for i, line in enumerate(m):
                # Get the words for each row
                words = split_line(line.lower())
                # Put the words in each row into the list
                all_words += words

    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    list = []
    # Stop words which create by myself
    stop_word =['the', 'in', 'to', 'of', 'and', 'has', 'have', 'that', 'The', 'for', 'on', 'is', 'at', 'are', 'by',
               'you', 'my', 'will', 'an', 'about']
    for item in list_to_remove:
        if not item.isalpha():
            list.append(item)
        elif len(item) == 1:
            list.append(item)
        elif item in stop_word:
            list.append(item)

    for l in list:
        del dictionary[l]

    dictionary = dictionary.most_common(3000)

    return dictionary


# Extract features over Bernoulli distribution
def extract_features(train_dir, datas):
    files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    features_matrix = []
    for fi in files:
        features_array = [0 for i in range(len(datas))]
        with open(fi) as f:
            for line in f:
                words = split_line(line.lower())
                for word in words:
                    for i, data in enumerate(datas):
                        if word == data:
                            features_array[i] = 1
        features_matrix.append(features_array)
    return features_matrix


# Extract features for Multinomial event models
def extract_features_e(train_dir, datas):
    files = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    features_matrix = []
    for fi in files:
        features_array = [0 for i in range(len(datas))]
        with open(fi) as f:
            for line in f:
                words = split_line(line.lower())
                for word in words:
                    for i, data in enumerate(datas):
                        if word == data:
                            features_array[i] += 1
        features_matrix.append(features_array)
    return features_matrix


# Extracting features of test data which will apply into Multi-variate Bernoulli model
def extract_features_of_test_data(test_dir, datas):
    features_array = [0 for i in range(len(datas))]
    with open(test_dir) as f:
        for line in f:
            words = split_line(line.lower())
            for word in words:
                for i, data in enumerate(datas):
                    if word == data:
                        features_array[i] = 1
    return features_array


# Label different type of class
def separate_class(datas):
    labeled_data = {}
    for i in range(3):
        labeled_data[i] = []
    for i in range(len(datas)):
        if i <= 4:
            labeled_data[0].append(datas[i])
        elif i <= 9:
            labeled_data[1].append(datas[i])
        else:
            labeled_data[2].append(datas[i])

    return labeled_data


