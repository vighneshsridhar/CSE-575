import numpy
import scipy.io
import math
import geneNewData

def main():
    myID='9999'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    #print([len(train0),len(train1),len(test0),len(test1)])
    #print('Your trainset and testset are generated successfully!')
    digit0_train = numpy.ndarray(shape = (len(train0), 2))
    digit1_train = numpy.ndarray(shape = (len(train1), 2))
    digit0_test = numpy.ndarray(shape = (len(test0), 2))
    digit1_test = numpy.ndarray(shape = (len(test1), 2))

    compute_mean_std_dev(train0, digit0_train)
    compute_mean_std_dev(train1, digit1_train)
    compute_mean_std_dev(test0, digit0_test)
    compute_mean_std_dev(test1, digit1_test)

    mean_feature1_digit0 = numpy.mean(digit0_train[:,0])
    variance_feature1_digit0 = numpy.var(digit0_train[:,0])
    mean_feature2_digit0 = numpy.mean(digit0_train[:,1])
    variance_feature2_digit0 = numpy.var(digit0_train[:,1])
    mean_feature1_digit1 = numpy.mean(digit1_train[:,0])
    variance_feature1_digit1 = numpy.var(digit1_train[:,0])
    mean_feature2_digit1 = numpy.mean(digit1_train[:,1])
    variance_feature2_digit1 = numpy.var(digit1_train[:,1])
    classify_test_digits(digit0_test, digit1_test, mean_feature1_digit0, variance_feature1_digit0, mean_feature2_digit0,
                         variance_feature2_digit0, mean_feature1_digit1, variance_feature1_digit1,
                         mean_feature2_digit1, variance_feature2_digit1)
    pass



if __name__ == '__main__':
    main()


def compute_mean_std_dev(data, output):
    for i, image in enumerate(data):
        output[i, 0] = numpy.mean(image)
        output[i, 1] = numpy.std(image)
    return output

def gaussian(mean, variance, data):
    gaussian = (1/(2 * math.pi * variance)**0.5) * math.exp((-1/(2 * variance)) * (data - mean)**2)
    return gaussian

def classify_test_digits(digit0_test, digit1_test, mean_feature1_digit0, variance_feature1_digit0, mean_feature2_digit0,
                         variance_feature2_digit0, mean_feature1_digit1, variance_feature1_digit1,
                         mean_feature2_digit1, variance_feature2_digit1):
    mean_digit0 = numpy.ndarray(shape = (len(digit0_test), 2))
    std_dev_digit0 = numpy.ndarray(shape = (len(digit0_test), 2))
    mean_digit1 = numpy.ndarray(shape = (len(digit1_test), 2))
    std_dev_digit1 = numpy.ndarray(shape = (len(digit1_test), 2))
    print(mean_feature1_digit0)
    print(variance_feature1_digit0)
    print(mean_feature2_digit0)
    print(variance_feature2_digit0)
    print(mean_feature1_digit1)
    print(variance_feature1_digit1)
    print(mean_feature2_digit1)
    print(variance_feature2_digit1)

    for i, x in enumerate (digit0_test[:, 0]):
        mean_digit0[i, 0] = gaussian(mean_feature1_digit0, variance_feature1_digit0, x)
        mean_digit0[i, 1] = gaussian(mean_feature1_digit1, variance_feature1_digit1, x)

    for i, x in enumerate(digit0_test[:, 1]):
        std_dev_digit0[i, 0] = gaussian(mean_feature2_digit0, variance_feature2_digit0, x)
        std_dev_digit0[i, 1] = gaussian(mean_feature2_digit1, variance_feature2_digit1, x)

    for i, x in enumerate(digit1_test[:, 0]):
        mean_digit1[i, 0] = gaussian(mean_feature1_digit0, variance_feature1_digit0, x)
        mean_digit1[i, 1] = gaussian(mean_feature1_digit1, variance_feature1_digit1, x)

    for i, x in enumerate(digit1_test[:, 1]):
        std_dev_digit1[i, 0] = gaussian(mean_feature2_digit0, variance_feature2_digit0, x)
        std_dev_digit1[i, 1] = gaussian(mean_feature2_digit1, variance_feature2_digit1, x)

    posterior0_digit0 = 0.5 * mean_digit0[:, 0] * std_dev_digit0[:, 0]
    posterior0_digit1 = 0.5 * mean_digit0[:, 1] * std_dev_digit0[:, 1]

    posterior1_digit0 = 0.5 * mean_digit1[:, 0] * std_dev_digit1[:, 0]
    posterior1_digit1 = 0.5 * mean_digit1[:, 1] * std_dev_digit1[:, 1]

    predicted_class_digit0 = posterior0_digit0 > posterior0_digit1
    predicted_class_digit1 = posterior1_digit1 > posterior1_digit0
    accuracy_digit0 = predicted_class_digit0.sum()/len(predicted_class_digit0)
    accuracy_digit1 = predicted_class_digit1.sum()/len(predicted_class_digit1)
    print(accuracy_digit0)
    print(accuracy_digit1)
