import math
import create_dictionary


# Calculate the fraction of a specific class to total class
def class_pro(data, label):
    probability_of_data = 0

    for keys in data.keys():
        probability_of_data += len(data[keys])

    probability_of_data = len(data[label])/probability_of_data

    return probability_of_data


# Calculate the posterior probability Without Laplace Smoothing
def posterior_pro(datas, label):
    L = []
    total_nums = 0

    for data in datas[label]:
        total_nums += sum(data)
    print(total_nums)
    for i in range(len(datas[label][0])):
        nums = 0
        for data in datas[label]:
            if data[i]:
                nums += data[i]

        L.append(nums / total_nums)

    return L


# Calculate the posterior probability with Laplace Smoothing
def posterior_pro_la(datas, label):
    L = []
    total_nums = 0

    for data in datas[label]:
        total_nums += sum(data)

    for i in range(len(datas[label][0])):
        nums = 0
        for data in datas[label]:
            if data[i]:
                nums += data[i]

        L.append((nums+1) / (total_nums + len(datas[label][0])))

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
    data_sorted = sorted(data)
    # print(data)

    # Extract features
    train_dir = "D:\Machine Learning\\training_datas"
    data = create_dictionary.extract_features_e(train_dir, data_sorted)
    labeled_data = create_dictionary.separate_class(data)
    class_pro = class_pro(labeled_data, 0)

    # Without Laplace smoothing
    medical = posterior_pro(labeled_data, 0)
    music = posterior_pro(labeled_data, 1)
    sport = posterior_pro(labeled_data, 2)

    # Using Laplace smoothing
    medical2 = posterior_pro_la(labeled_data, 0)
    music2 = posterior_pro_la(labeled_data, 1)
    sport2 = posterior_pro_la(labeled_data, 2)

    # Loading test data and test the data1
    test_dir = "D:\Machine Learning\\test_data\medical6.txt"
    test_data1 = create_dictionary.extract_features_of_test_data(test_dir, data_sorted)
    M = probability_of_one_label(test_data1, medical2, class_pro)
    S = probability_of_one_label(test_data1, sport2, class_pro)
    Mu = probability_of_one_label(test_data1, music2, class_pro)
    print("Log of probability of medical, music and sport are {0}, {1} and {2}, respectively".format(M, Mu, S))

    # Loading test data and test the data2
    test_dir2 = "D:\Machine Learning\\test_data\music7.txt"
    test_data2 = create_dictionary.extract_features_of_test_data(test_dir2, data_sorted)
    M = probability_of_one_label(test_data2, medical2, class_pro)
    S = probability_of_one_label(test_data2, sport2, class_pro)
    Mu = probability_of_one_label(test_data2, music2, class_pro)
    print("Log of probability of medical, music and sport are {0}, {1} and {2}, respectively".format(M, Mu, S))

    # Loading test data and test the data3
    test_dir3 = "D:\Machine Learning\\test_data\sport6.txt"
    test_data3 = create_dictionary.extract_features_of_test_data(test_dir3, data_sorted)
    M = probability_of_one_label(test_data3, medical2, class_pro)
    S = probability_of_one_label(test_data3, sport2, class_pro)
    Mu = probability_of_one_label(test_data3, music2, class_pro)
    print("Log of probability of medical, music and sport are {0}, {1} and {2}, respectively".format(M, Mu, S))