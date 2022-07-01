"""
@author: Muamer Hrncic, Daniel Holzfeind, Alexandra Wissiak
license: MIT
"""
import os
import numpy as np
import cv2 as cv

def get_HOG_descriptor(winSize=100, blockSize=10, blockStride=5, cellSize=10, nbins=9, derivAperture=1,
                       winSigma=-1., histogramNormType=0, L2HysThreshold=0.2, gammaCorrection=1, nlevels=64,
                       signedGradient=True):
    """
    Method which returns the open cv implementation of the HOG descriptor.

    Attributes
    ----------
    winSize: int, optional
        Sets winSize with given value.
    blockSize: int, optional
        Sets blockSize with given value.
    blockStride: int, optional
        Sets blockStride with given value.
    cellSize: int, optional
        Sets cellSize with given value.
    nbins: int, optional
        Sets nbins with given value.
    derivAperture: int, optional
        Sets nbins with given value.        
    winSigma: double, optional
        Sets nbins with given value.        
    histogramNormType: int, optional
        Sets nbins with given value.        
    L2HysThreshold: double, optional
        Sets nbins with given value.    
    gammaCorrection: int, optional
        Sets nbins with given value.        
    nlevels: int, optional
        Sets nbins with given value.        
    signedGradient: boolean, optional
        Sets nbins with given value.
        
    Returns
    -------
        cv.HOGDescriptor(params)
    """

    return cv.HOGDescriptor((winSize, winSize), (blockSize, blockSize), (blockStride, blockStride), (cellSize, cellSize),
                            nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

def load_data(data_path):
    """
    Method to get either paths to training or test data and the class labels.

    Attributes
    ----------
    data_path: string
        Path to all images
        
    Returns
    -------
    data:
        Paths to images of the components. 
    labels:
        Label classe of every image.
    """
    #List to hold the image paths
    data = []
    #List to hold the labels
    labels = []
    #Label index
    i = 0
    #Sort folders
    directories = sorted(os.listdir(data_path))
    
    #Iterate over all paths
    for directory in directories:
        #Store all file names in the current folder
        files = os.listdir(data_path + '/' + directory)
        #Search for image files
        for file in files:
            if("png" in file or "jpg" in file or "jpeg" in file):
                #Append the path to the image
                data.append(data_path + '/' + directory + '/' + file)
                #Append the actual label
                labels.extend([x for x in np.repeat(i, 4)])
        #Print the actual class
        print("{} -> {} ".format(directory, i))
        #increment label
        i = i+1

    return data, labels

def train_component_SVM(train, labels, resize = 100, kernelsize = 9, blocksize = 11, const = 1):
    """
    Method to create and train a MC-SVM.

    Attributes
    ----------
    train: list,
        Paths to the training data.
        
    labels: list
        Labels of the training data.
        
    resize: int, optional
        Size of the resized image.
        
    kernelsize: int, optional
        Size of the kernel used for Gaussian blurring
        
    blocksize: int, optional
        Size of blocks for adaptive thresholding.
        
    const: int, optional
        Constant for adaptive thresholding. 
        
    Returns
    -------
    svm:
        Trained SVM object. 
    """
    #Create a Hog descriptor
    hog = get_HOG_descriptor();
    #List to hold the calculated HOG descriptors
    descriptors = []
    #Create an instance of a support vector machine
    svm = cv.ml.SVM_create()
    #Set kernel type to radial basis function
    svm.setKernel(cv.ml.SVM_RBF)
    #Set SVM type as C-Support Vector Classification type. 
    #n-class classification (n ≥ 2), allows imperfect separation 
    #of classes with penalty multiplier C for outliers. 
    svm.setType(cv.ml.SVM_C_SVC)
    
    #Actual image
    act_img = 0
    #Collect train data
    for path in train:
        #Read image as Grayscale image
        img = cv.imread(path, 0)
        #Resize the image to get a usefull amount of features
        resized = cv.resize(img, (resize,resize), interpolation = cv.INTER_CUBIC)
        #Blur the image to reduce noise
        blurred = cv.GaussianBlur(resized, (kernelsize,kernelsize) , 0)
        #Apply adaptive thresholding to the blurred image
        threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blocksize, const)
        #Save the threshold image
        cv.imwrite('data/00_threshold_img/' + str(act_img) + '.jpg', threshold)
        #Compute the hog descriptors and append them to the overall list
        descriptors.append(hog.compute(threshold))
        #Get image shape
        rows,cols = resized.shape
        #Rotate the image three time by 90 degrees and repeat above process.
        #This ensures kind of a rotation invariance in the case of normal circuits.
        #This means that components are aligned either vertical or horizontal.
        for i in [1,2,3]:
            #Get the rotation matrix which rotates the picture by i*90 degrees, around the image center
            #and keep the original scale
            rotmatrix  = cv.getRotationMatrix2D((cols/2,rows/2), i*90, 1)
            #Save the rotated picture
            rotated = cv.warpAffine(threshold, rotmatrix, (cols,rows))
            #Compute the hog descriptors of the rotated image and append them to the overall list
            descriptors.append(hog.compute(rotated))
            #Store the image
            cv.imwrite('data/00_threshold_img/'+str(act_img)+'_'+str(i)+'.jpg', rotated)
        #Increase image count
        act_img += 1

    print("Threshold pictures are created.")
    print("SVM-Training starts now.")
    svm.trainAuto(np.array(descriptors, np.float32), cv.ml.ROW_SAMPLE, np.array(labels))
    print("SVM is trained.")
    print("SVM data will be stored to: data/02_svm/trained.dat")
    svm.save('data/02_svm/trained.dat')    
    
    return svm

def test_component_SVM(test, labels, resize = 100, kernelsize = 9, blocksize = 11, const = 2):
    """
    Method to test a MC-SVM.

    Attributes
    ----------
    test: list,
        Paths to the test data.
        
    labels: list
        Labels of the training data.
        
    resize: int, optional
        Size of the resized image.
        
    kernelsize: int, optional
        Size of the kernel used for Gaussian blurring
        
    blocksize: int, optional
        Size of blocks for adaptive thresholding.
        
    const: int, optional
        Constant for adaptive thresholding. 
        
    Returns
    -------
    class_corr:
        List with number of correctly classified components
    class_total:
        List with total number of classified components
    """
    #Create a Hog descriptor
    hog = get_HOG_descriptor();
    #List to hold the calculated HOG descriptors
    descriptors = []
    #Load the data of the trained SVM
    svm = cv.ml.SVM_load("data/02_svm/trained.dat")
    
    #Actual image
    act_img = 0
    #Collect test data
    for path in test:
        #Read image as Grayscale image
        img = cv.imread(path, 0)
        #Resize the image to get a usefull amount of features
        resized = cv.resize(img, (resize,resize), interpolation = cv.INTER_CUBIC)
        #Blur the image to reduce noise
        blurred = cv.GaussianBlur(resized, (kernelsize,kernelsize) , 0)
        #Apply adaptive thresholding to the blurred image
        threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blocksize, const)
        #Compute the hog descriptors and append them to the overall list
        descriptors.append(hog.compute(threshold))
        #Get image shape
        rows,cols = resized.shape
        #Rotate the image three time by 90 degrees and repeat above process.
        #This ensures kind of a rotation invariance in the case of normal circuits.
        #This means that components are aligned either vertical or horizontal.
        for i in [1,2,3]:
            #Get the rotation matrix which rotates the picture by i*90 degrees, around the image center
            #and keep the original scale
            rotmatrix  = cv.getRotationMatrix2D((cols/2,rows/2), i*90, 1)
            #Save the rotated picture
            rotated = cv.warpAffine(threshold, rotmatrix, (cols,rows))
            #Compute the hog descriptors of the rotated image and append them to the overall list
            descriptors.append(hog.compute(rotated))
        #Increase image count
        act_img += 1
    #Init List of correctly classified components
    class_corr = [0, 0, 0]
    #Init List of all classified components
    class_total = [0, 0, 0]

    #Store test data in a suitable format for classification and analysis
    test_data = zip(descriptors, labels, range(1, len(labels) + 1))
    #Classify all components
    for (descriptor_i, true_label, i) in zip(descriptors, labels, range(1, len(labels) + 1)):
        #Predict the class of the actual component and store the predicted class
        _, class_hat = svm.predict(np.array(descriptor_i, np.float32).reshape(-1, svm.getVarCount()))
        #Convert the predicted class to integer
        pred_comp = int(class_hat[0][0])
        #If the predicted class equals the true class
        if(pred_comp == true_label):
            #increase the counter for true positives
            class_corr[true_label] += 1
        #Increase the counter of the total predicted components
        class_total[pred_comp] += 1
    return class_corr, class_total

def classify(img, rects, boxes):
    """
    Method to classify diodes, resistors and inductors.

    Attributes
    ----------
    img: numpy.array
        Image to search for components. Optimally threshold image of the reduced circuit.
        
    rects: numpy.array
        Bounding rectangles of components. Serve as region of interest.
        
    boxes: list
        List to hold the classified components.

    Returns
    -------
    boxes: list
        List to hold the classified components.
    """
    #Load data of the SVM
    svm = cv.ml.SVM_load("data/02_svm/trained.dat")
    #Create HOG descriptor for feature detection
    hog = get_HOG_descriptor();
    #Save number of classes
    nrClasses = svm.getTermCriteria()[0]
    #Save number of variable
    nrVars = svm.getVarCount()
    
    #Iterate over all reactangles
    for xs, ys, xe, ye in rects:
        #Get region of interest and resize the image to 100x100, using cubic interpolation
        region = cv.resize(img[ys : ye, xs : xe], (100,100),interpolation = cv.INTER_CUBIC)
        #Compute HOG features
        descriptors = hog.compute(region)
        #Predict the class of the component
        _, result = svm.predict(np.array(descriptors, np.float32).reshape(-1,nrVars))
        #Store predicted class
        cl = int(result[0][0]) + nrClasses
        boxes.append([[int(xs), int(ys), int(xe - xs),int(ye - ys)],cl])
    return boxes

def get_component_SVM():
    """
    Method to get the trained SVM.

    Returns
    -------
    SVM:
        Trained SVM.

    """
    #Load the data of the trained SVM and return it
    return cv.ml.SVM_load("data/02_svm/trained.dat") 