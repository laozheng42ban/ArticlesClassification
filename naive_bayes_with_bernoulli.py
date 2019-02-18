import math
import create_dictionary


# Calculate the fraction of a specific class to total class
def class_probability(data, label):
    probability_of_data = 0

    for keys in data.keys():
        probability_of_data += len(data[keys])

    probability_of_data = len(data[label])/probability_of_data

    return probability_of_data


# Calculate the posterior probability Without Laplace Smoothing
def posterior_pro(datas, label):

    L = []
    for i in range(len(datas[label][0])):
        nums = 0
        for data in datas[label]:
            if data[i] == 1:
                nums += 1

        L.append(nums / 5)

    return L


# With Laplace smoothing
def posterior_pro_la(datas, label):

    L = []
    for i in range(len(datas[label][0])):
        nums = 0
        for data in datas[label]:
            if data[i] == 1:
                nums += 1

        L.append((nums+1) / (5 + 3))

    return L


# Predicting the class prior
def probability_of_one_label(test_datas, train_data, class_probability):
    probability = 1
    for i,test_data in enumerate(test_datas):
        if test_data == 0:
            probability += math.log(1-train_data[i])
        else:
            probability += math.log(train_data[i])

    return probability + math.log(class_probability)


if __name__ == "__main__":
    # Directory of dictionary
    dict_dir = "D:\Machine Learning\dict"
    dictionary = create_dictionary.make_dictionary(dict_dir)

    data = []
    # Sorting the datasets
    for sp in dictionary:
        data.append(sp[0])
    data1 = sorted(data)

    # Extract features from training data
    train_dir = "D:\Machine Learning\\training_datas"
    data = create_dictionary.extract_features(train_dir, data1)
    labeled_data = create_dictionary.separate_class(data)

    # Calculate the log maximum likelihood estimate
    class_pro = class_probability(labeled_data, 0)
    medical = posterior_pro_la(labeled_data, 0)
    music = posterior_pro_la(labeled_data, 1)
    sport = posterior_pro_la(labeled_data, 2)

    # Loading test data and test the data1
    test_dir = "D:\Machine Learning\\test_data\medical6.txt"
    test_data1 = create_dictionary.extract_features_of_test_data(test_dir, data1)
    M = probability_of_one_label(test_data1, medical, class_pro)
    S = probability_of_one_label(test_data1, sport, class_pro)
    Mu = probability_of_one_label(test_data1, music, class_pro)
    print("Log of probability of medical, music and sport are {0}, {1} and {2}, respectively".format(M, Mu, S))

    # Loading test data and test the data2
    test_dir2 = "D:\Machine Learning\\test_data\music7.txt"
    test_data2 = create_dictionary.extract_features_of_test_data(test_dir2, data1)
    M = probability_of_one_label(test_data2, medical, class_pro)
    S = probability_of_one_label(test_data2, sport, class_pro)
    Mu = probability_of_one_label(test_data2, music, class_pro)
    print("Log of probability of medical, music and sport are {0}, {1} and {2}, respectively".format(M, Mu, S))

    # Loading test data and test the data3
    test_dir3 = "D:\Machine Learning\\test_data\sport6.txt"
    test_data3 = create_dictionary.extract_features_of_test_data(test_dir3, data1)
    M = probability_of_one_label(test_data3, medical, class_pro)
    S = probability_of_one_label(test_data3, sport, class_pro)
    Mu = probability_of_one_label(test_data3, music, class_pro)
    print("Log of probability of medical, music and sport are {0}, {1} and {2}, respectively".format(M, Mu, S))

