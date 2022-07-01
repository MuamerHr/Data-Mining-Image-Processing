"""
@author: Muamer Hrncic, Daniel Holzfeind, Alexandra Wissiak
license: MIT
"""
import cv2 as cv
import numpy as np
from electrical_comp_SVM import load_data, train_component_SVM, get_HOG_descriptor

if __name__ == '__main__':
    
    #Path to the train data
    train_path = "data/01_train"
    #Load the train data and the labels
    train, labels  = load_data(train_path)
    #Train the SVM. Caution! This takes a while and demands a lot of ressources.
    comp_svm = train_component_SVM(train, labels)
    #Test the trained SVM with a test image
    test = cv.imread("data/circuit.png",0)
    #Create a HOG descriptor
    hog = get_HOG_descriptor();
    #Compute the HOG descriptors of the test image
    test_hog = hog.compute(test)
    #Classify the components
    re = comp_svm.predict(np.array(test_hog ,np.float32).reshape(-1, comp_svm.getVarCount()))
    #Print the result of the classification
    print(re)
