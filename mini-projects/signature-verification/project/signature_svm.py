# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@author:  Freislich, Hrncic, Siebenhofer

to use the code: pip install imagehash, pip install opencv-python
License: MIT
"""


import numpy as np
import cv2 as cv
import imagehash
import pickle
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import datetime

class signature_svm(object):
    """
    This class creates and trains a support vector machine to validate signatures.
    
    Parameters
    ----------
    ----------
        
    """
    ##########################################################################
    #Init class data.
    def __init__(self, n_clusters=120, random_state=None, write_output = True, penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, max_iter=1000):
        #Number of users with signatures
        self.nr_users = 0
        #Number of clusters for the kmeans algorithm
        self.n_clusters = n_clusters
        #Kmeans object used for bag of visual words
        self.kmeans_obj = KMeans(n_clusters=n_clusters, random_state=random_state)
        #Predicted labels after fitting with kmeans
        self.kmeans_labels = None
        #Vocabulary histogram
        self.histogram = None
        #List of lists of descriptors
        self.descriptor_list = []
        #Stacked SIFT-Descriptors of the signature images
        self.descriptors = None
        #Number of images per user in the signature "library"
        self.nr_images_in_lib = 0
        #Labels of the signatures (valid, invalid) = (1, 2)
        self.labels = []
        #List of the image contour features
        self.img_feat_cont = []
        #Linear Support Vector Classifier/Machine for classifying signatures
        self.clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class, fit_intercept=fit_intercept,
                             intercept_scaling=intercept_scaling, class_weight=class_weight, verbose=verbose, random_state=random_state, max_iter=max_iter)
        #Indicates wether self.clf is trained or not
        self.trained = False
        self.write_output = write_output
        self.output_path = "output"
        
    ##########################################################################
    #Image processing methods.  Morphological Operations and Feature
    #extraction.
    def get_sift_features(self, preprocessed):
        #Create SIFT-feature detector
        sift = cv.SIFT_create()
        #Detect keypoints and compute descriptors
        _, descriptors = sift.detectAndCompute(preprocessed, None)
        #Return descriptors
        return descriptors

    def get_img_contour_features(self, img):
        #Get the the minimum area rotated rectangle of the signature.
        rect = cv.minAreaRect(cv.findNonZero(img))
        
        #Get the box point of the rectangle
        box = cv.boxPoints(rect)
        box = np.int0(box)
        
        #Calculate the width and the height of the rectangle
        w = np.linalg.norm(box[0] - box[1])
        h = np.linalg.norm(box[1] - box[2])
        
        #Calculate the aspect ratio
        aspect = max(w, h) / min(w, h)
        #Calculate the are of the bounding rectangle
        area_bnd_rect = w * h
        
        #Get the convex hull of the image
        conv_hull = cv.convexHull(cv.findNonZero(img))
        #Search for contours
        contours, _ = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        #Variable to hold the total area of all contours
        area_cont = 0

        #Sum up the area of all contours
        for cnt in contours:
            area_cont += cv.contourArea(cnt)
        #Store the area of the convex hull
        area_conv_hull = cv.contourArea(conv_hull)

        #Calculate the area ratio between the convex hull of the image and the
        #minimum area rotated rectangle
        ratio_conv_hull_bnd_rect = area_conv_hull / area_bnd_rect
        #Calculate the area ratio betwenn the contours and the minimum area
        #rotated rectangle
        ratio_cont_bnd_rect = area_cont / area_bnd_rect
        
        #Return all values, they will also be used as features
        return aspect, ratio_conv_hull_bnd_rect, ratio_cont_bnd_rect

    def compute_img_features(self, image_path, feat_dict, thin = True):
        #Preprocess the image
        preproc = self.preprocess_img(image_path, thin = thin)
        #Calculate the imagehash.  Image hashes tell whether two images look
        #nearly identical.
        hash_val = imagehash.phash(Image.open(image_path))
        #Get the additional image features
        aspect, ratio_conv_hull_bnd_rect, ratio_cont_bnd_rect = self.get_img_contour_features(preproc.copy())
        #Convert the hash value
        hash_val = int(str(hash_val), 16)
        #Append the calculated values to the actual image dictionary
        feat_dict['hash'] = hash_val
        feat_dict['aspect_ratio'] = aspect
        feat_dict['hull_bounding_rect_ratio'] = ratio_conv_hull_bnd_rect
        feat_dict['contour_bounding_ratio'] = ratio_cont_bnd_rect
        
        #Store the features in a separate list
        features = [hash_val, aspect,
                    ratio_conv_hull_bnd_rect, ratio_cont_bnd_rect]
        #Get SIFT descriptors
        sift_feat = self.get_sift_features(preproc)
        
        #Return all values
        return features, sift_feat

    def get_img_feat(self, feat_dict, path, img_feat, thin = True):
        #Create the path to the actual image
        image_path = path + "/" + feat_dict['name']
        #Get all features of the image
        features, sift_feat = self.compute_img_features(image_path, feat_dict, thin = thin)
        #Append the list of the four additional features to the global list
        self.img_feat_cont.append(features)
        #Append the SIFT descriptors to the global list
        self.descriptor_list.append(sift_feat)

    def get_features_and_targets(self, img_feat_val, img_feat_inval, path_valid="signatures/valid", path_invalid="signatures/invalid", thin = True):

        #Iterate over all user
        for i in range(self.nr_users):
            
            #Store the number of valid signatures for the actual user
            nrval = len(img_feat_val[i])
            #Store the number of valid signatures for the actual user
            nrinval = len(img_feat_inval[i])
            
            #Get the feature of all valid signatures
            for feat_dict in img_feat_val[i]:
                self.get_img_feat(feat_dict, path_valid, img_feat_val[i], thin = thin)
            
            #Get the feature of all invalid signatures
            for feat_dict in img_feat_inval[i]:
                self.get_img_feat(feat_dict, path_invalid, img_feat_inval[i], thin = thin)
            
            #Increase the total number of images in the signature library
            self.nr_images_in_lib += nrval + nrinval
            #Extend the label/class/target vector, 1 is valid, 2 is invalid
            self.labels.append([1 for x in range(nrval)] + [2 for x in range(nrinval)])
        
        #Reformat the descriptor list
        _ = self.reformat_descriptor_list(self.descriptor_list)
        
        #Perform bag of visual words algorithm; Cluster SIFT descriptors with
        #kmeans. The center of each cluster will be used as the visual dictionary’s
        #vocabulary.
        self.cluster_descriptors()
        #Create the vocabulary histogram to get the feature vectors for every
        #image
        self.create_vocab_hist(nr_images=self.nr_images_in_lib, descriptors=self.descriptor_list)
        
        #Initialize a temporary matrix with zeros
        extend_hist = np.zeros((self.nr_images_in_lib, self.n_clusters + 4), np.float32)
        
        #Store the number of columns of the matrix
        _, n = extend_hist.shape

        #Copy the vocabulary histogramm to the temporary matrix and also add
        #the
        #4 additional features in the last 4 columns
        extend_hist[:, 0: n - 4] = self.histogram[:][:]
        extend_hist[:, n - 4: n] = self.img_feat_cont[:][:]
        
        #Update the original histogram
        self.histogram = extend_hist.copy()
        #Standardize histogram entries
        self.standardize()

        #Reshape target/label vector
        self.labels = np.array([item for sublist in self.labels for item in sublist], np.float32)

        #Return copies of the histogram and the labels which will
        #act as feature matrix and target for the linear SVC
        return self.histogram.copy(), self.labels.copy()
    
    def create_outputFolder(self,path):
        # creating folder in output named after the current time
        datestring = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
        direct = "output/" + datestring
        # create folder if not already exists
        if not os.path.exists(direct):
            os.mkdir(direct)
        self.output_path = direct
        return direct
    
    def preprocess_img(self, path, thresh=30, maxval=255, threstype=0, thin = True):
        # preprocessng the masked image
        raw_image = cv.imread(path)
        # Get the grayscale image
        bw_image = cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY)
        #Invert colors
        bw_image = 255 - bw_image
        #Apply thresholding to the grayscale image
        _, threshold = cv.threshold(bw_image, thresh, maxval, threstype)
        # crop the image to readuce cost of following operations
        cropped = self.crop_image(threshold)
        # use with different parameters to check if it has influence on result
        if(thin):
            thinned = self.thinning(cropped)
        else:
            thinned = cropped.copy()
        
        #Check wether to write output
        if self.write_output:
            # create output folder and get directory
            directory = self.create_outputFolder(path)
            # get the filename from the given path
            fname = ""
            st = path.split("/")[-1].split(".")[:-1]
            for ele in st:
                fname += ele
            
            # write the files into generated output folder
            cv.imwrite(directory + "/" + fname + "_black_and_white.png", bw_image)
            cv.imwrite(directory + "/" + fname + "_threshold.png", threshold)
            cv.imwrite(directory + "/" + fname + "_thinned.png", thinned)    
            cv.imwrite(directory + "/" + fname + "_cropped.png", cropped)
            
        #return threshold
        return thinned

    def crop_image(self, img, tol=0):
        #Indices where image pixel have value greate than zero
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def crop_and_thinn(self, path, kernelsize=5, tol=0):
        #path = "signatures/valid/001001_001.png"
        img = cv.imread(path, 0)
        #cv.imshow("input",img)

        # binarization via gaussianblur and otsu thresholding

        blur = cv.GaussianBlur(img, (kernelsize, kernelsize), 0)
        _, thres = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        image = np.invert(thres)
        #cv.imshow("binarized",image)

        cropped = self.crop_image(np.invert(thres), tol=tol)
        #cv.imshow("cropped",cimg)
        
        #Thin the cropped image
        thinned = self.thinning(cropped)
        #Show the preprocessing steps
        fig = plt.figure(num=1, figsize=(12, 6))
        plt.subplot(141), plt.imshow(img, cmap='gray')
        plt.title("original image")
        plt.subplot(142), plt.imshow(image, cmap='gray')
        plt.title("binarized image")
        plt.subplot(143), plt.imshow(cropped, cmap='gray')
        plt.title("cropped image")
        plt.subplot(144), plt.imshow(thinned, cmap='gray')
        plt.title("thinned image")
        fig.savefig("output/crop_and_thinn.png")

    def thinning(self, img, kernelsize=3):
        #Total number of entries of the image matrix, nrrow * nrcol
        size = np.size(img)
        #Initialize the resulting image with zeros
        skel = np.zeros(img.shape, np.uint8)
        #ret, img = cv.threshold(img, thresh, maxval, threstype)
        #Get structuring element for morphological operations
        element = cv.getStructuringElement(cv.MORPH_CROSS, (kernelsize, kernelsize))

        #Thinning loop
        while(True):
            #Erode the image
            eroded = cv.erode(img, element)
            #dilate the eroded image
            temp = cv.dilate(eroded, element)
            #Subtract the opened image from the original
            temp = cv.subtract(img, temp)
            #Get the skeleton through bitwise OR of the skeleton
            #and the temporary image
            skel = cv.bitwise_or(skel, temp)
            #Make a copy of the first result
            img = eroded.copy()
            #and check if there is nothing more to thin
            zeros = size - cv.countNonZero(img)
            #If so, stop the algorithm, and otherwise continue
            if zeros == size:
                break
        return skel

    ##########################################################################
    # Bag of Visual Words:
    # Kmeans Clustering of SIFT - Descriptors,
    # Vocabulary histogram / Feature creation
    def get_img_library_Data(self, path_valid="signatures/valid", path_invalid="signatures/invalid", thin = True):
        #Get all valid signature images
        valid = listdir(path_valid)
        #Get all invalid signature images
        invalid = listdir(path_invalid)
        #Initialize the list of found identification numbers
        id_list = []
        
        #Append any found id
        for fname in valid:
            id_list.append(int((fname.split("_")[0][-3:])) - 1)
        #The number of unique ids is the number of total users, store it
        self.nr_users = len(set(id_list))
        
        #Initialize the list which holds the valid image per user, per image
        img_feat_val = [[]for x in range(self.nr_users)]
        #Initialize the list which holds the invalid image per user, per image
        img_feat_inval = [[]for x in range(self.nr_users)]

        for fname in valid:
            #extract the signature/user id
            sig_id = int((fname.split("_")[0][-3:])) - 1
            #append the valid signature to the dictionary of the user
            #with the extracted id
            img_feat_val[sig_id].append({"name": fname})

        for fname in invalid:
            #extract the signature/user id
            sig_id = int((fname.split("_")[0][-3:])) - 1
            #append the valid signature to the dictionary of the user
            #with the extracted id
            img_feat_inval[sig_id].append({"name": fname})

        return self.get_features_and_targets(img_feat_val, img_feat_inval, path_valid, path_invalid, thin = thin)

    def cluster_descriptors(self):

        #Fit k-means clustering of the found SIFT descriptors to get the
        #visual vocabulary and predict the labels.
        #The cluster centroids are the words/vocabularys.
        self.kmeans_labels = self.kmeans_obj.fit_predict(self.descriptors)

    def create_vocab_hist(self, nr_images, descriptors, kmeans_labels=None):
        #Initialize the histogram
        self.histogram = np.array([np.zeros(self.n_clusters)
                                  for i in range(nr_images)], np.float32)
        #Counter
        prev = 0
        for i in range(nr_images):
            #Save the length of the actual descriptor list
            n = len(descriptors[i])
            
            for j in range(n):
                #Save the predicted index
                act_ind = self.kmeans_labels[prev + j] if (kmeans_labels is None) else kmeans_labels[prev + j]
                #Increase the corresponding histogram entry
                self.histogram[i][act_ind] += 1
            #increase the counter
            prev += n

    def standardize(self, scaler=None):        
        if scaler is None:
            #Initialize the standard scaler
            self.scale = StandardScaler().fit(self.histogram)
            #Standardize the histogram to overcome bias
            self.histogram = self.scale.transform(self.histogram)
        else:
            self.histogram = scaler.transform(self.histogram)

    def reformat_descriptor_list(self, descriptor_list):
        #Get the first list
        descriptors = np.array(descriptor_list[0])
        #Stack all together into as vertical stack
        for descritor in descriptor_list[1:]:
            descriptors = np.vstack((descriptors, descritor))
        #Save the list to the global descriptor stack
        self.descriptors = descriptors.copy()
        return descriptors

    def plot_histogram(self):
        #Make a copy of the histogram
        hist = self.histogram.copy()
        #Create values for the x-axis of the plot
        x_scalar = np.arange(self.n_clusters)
        #Get the bar heights for all clusters
        y_scalar = np.array([abs(np.sum(hist[:, h], dtype=np.int32)) for h in range(self.n_clusters)])

        fig = plt.figure(num=1, figsize=(12, 8))
        #Create the histogram bars
        plt.bar(x_scalar, y_scalar)
        #Define axis labels
        plt.xlabel("Index")
        plt.ylabel("Occurencies")
        plt.title("Descriptor vocabulary histogram")
        plt.show()
        fig.savefig(self.output_path + '/histogram.jpg')
    
    ##########################################################################
    #Linear SVC:
    #Create Linear support vector classifier with vocabulary as features and
    #two classes target, (1,2) = ("valid","invalid")
    def fit(self, X_train, y_train):
        #Copy train features
        self.X_train = X_train.copy()
        #Copy train target
        self.y_train = y_train.copy()
        #Fit the model
        self.clf.fit(self.X_train, self.y_train)
        #Mark the model as trained
        self.trained = True

    def predict(self, X):
        #Predict class for feature X
        return self.clf.predict(X)

    def score(self, X_test, y_test):
        #Return mean accuracy of the test data set
        return self.clf.score(X_test, y_test)

    def get_clf_obj(self):
        #Return the classifier object
        return self.clf

    def save_svm(self, filename='svm/trained.sav'):
        #Save the signature_svm if it is trained.
        if(self.trained):
            #Define filename
            filename = 'svm/trained.sav'
            #Save Model Using Pickle
            pickle.dump(self, open(filename, 'wb'))
    
    ##########################################################################
    #Auxilliary methods for testing "new images" outside of the signature lib.

    def image_to_feature_vector(self, img_sig, thin = True):
        #Create empty list
        feat_dict = []
        #Append dictionary with the file name
        feat_dict.append({"name": img_sig})
        #Get feature of the image
        features, sift_feat = self.compute_img_features(img_sig, feat_dict[0], thin = thin)

        # generate vocabulary of descriptors for test image
        vocab = np.array([[0 for i in range(self.n_clusters + 4)]], "float")
        # locate nearest clusters for each of
        # the visual word (features) present in the image
        test_ret = self.kmeans_obj.predict(sift_feat)
        
        #Increase the image vocabulary for every predicted index
        for each in test_ret:
            vocab[0][each] += 1
        
        #Extend the vocabulary with the four additional feature
        vocab[0, self.n_clusters:] = features[:]
        #Standardize the feature vector
        return self.scale.transform(vocab)

    def check_signature(self, img_sig, thin = True):

        #Get the feature vector of the image
        X_img = self.image_to_feature_vector(img_sig, thin = thin)

        # predict the class of the image
        yhat = int(self.clf.predict(X_img))

        #If the predicted class is one the signature can be considered
        #as valid
        if(yhat == 1):
            return "valid"
        else:
            #otherwise it is invalid
            return "invalid"

    ##########################################################################
def load_svm_from_datafile(filename='svm/trained.sav'):
    # load the model from disk
    return pickle.load(open(filename, 'rb'))
