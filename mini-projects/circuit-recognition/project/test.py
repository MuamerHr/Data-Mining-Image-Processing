"""
@author: Muamer Hrncic, Daniel Holzfeind, Alexandra Wissiak
license: MIT
"""
from electrical_comp_SVM import load_data, test_component_SVM

if __name__ == '__main__':

    #Path to the test data
    image_path = "data/03_test"
    #Load the train data and the labels
    test, labels = load_data(image_path)
    #Number of test images per class
    nrLabels = len(labels)
    #Classify the test data images and get the correctly classified components
    #and the total classified components
    class_corr, class_total = test_component_SVM(test, labels)
    #Calculate the precision and the recall of the classification
    for i in range(len(class_corr)):
        #The precision for class i is the ratio of true positives for class i and total classified objects in class i
        precision = class_corr[i] / float(class_total[i])
        #The recall for class i is the ratio of true positives for class i and the total number of classes
        #which is 3 in this case, since the three line components are classified with other methods
        recall = class_corr[i] / float((nrLabels/3))
        #print the result
        print("Performance for class {} \nprecision :{:.3f}\trecall :{:.3f}".format(
            i, precision, recall,))
        print("Nr. of correctly classified {}\t Nr of test images {}".format(class_corr[i], int(nrLabels/3)))
