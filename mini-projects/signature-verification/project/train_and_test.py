# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@author:  Freislich, Hrncic, Siebenhofer

to use the code: pip install imagehash, pip install opencv-python
License: MIT
"""

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from signature_svm import signature_svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

def plot_act_score(x_axis, y_axis, clsize, startval = 230, nrsub = 1,):
    ax = fig4.add_subplot(startval + nrsub)
    #Draw test scores, 30 clusters
    ax.plot(x_axis, y_axis, marker = 'o', label = "Scores")
    #Add legend
    ax.legend(loc="lower right")
    #Add Title
    title = "Scores, cluster size " + str(clsize)
    ax.set_title(title)
    #Add y-axis label
    ax.set_ylabel("Mean score")

if __name__ == "__main__":
    ##########################################################################
    #Training once and simple testing 
    #Seed for random number generator
    randomCharSeed = 80827983   #or randomCharSeed = None
    #Train size
    size = 0.8
    
    #Create signature SVM with 120 clusters for feature clustering and a given random seed
    sigSVM = signature_svm(n_clusters=120, random_state = randomCharSeed, write_output=True)
    
    #Get the feature matrix and the target (labels) of all images in the signature folder
    X, y = sigSVM.get_img_library_Data()
    
    #Split the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=size, random_state=randomCharSeed)
    
    #Fit the model
    sigSVM.fit(X_train, y_train)
    
    #Compute score
    score = sigSVM.score(X_test, y_test)
    
    #Show Vocabulary Histogram
    sigSVM.plot_histogram()
    
    #predict classes of all images in the image library
    y_hat = sigSVM.predict(X)
    
    #Get classes
    clf_classes = np.array(sigSVM.get_clf_obj().classes_, int)
    #Create the confusion matrix from the result
    conf_matrix = confusion_matrix(y, y_hat, labels = clf_classes)

    #Create the confusion matrix
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=clf_classes)
    #Display the confusion matrix
    disp.plot()
    
    #save plot
    disp.figure_.savefig("testing/exhaustive_testing/confusion_matrix.png")

    #Get the entries of the confusion matrix, where
    #TP...True positives: Valid pictures classified as valid
    #TN...False positives: Valid pictures classified as invalid
    #FN...False negatives: Invalid pictures classified as valid
    #FP...False positives: Invalid pictures classified as invalid
    (TP, FP, FN, TN) = np.reshape(conf_matrix, 4)
    
    #Get performance metrics and print them 
    accuracy = (TP + TN) / (TP + TN + FN + FN)
    print("Accuracy: ", accuracy)
    specifity = TN / (TN + FP) 
    print("specifity: ", specifity)
    precision = TP / (TP + FP)
    print("precision: ", precision)
    recall = TP / (TP + FN)
    print("recall: ", recall)

    #Store metrics
    perf_metrics = [accuracy, specifity, precision, recall]
    #Create column names for a table
    colNames = ["accuracy", "specifity", "precision", "recall"]
    
    #Create subplot
    fig2, ax = plt.subplots()

    # hide axes
    fig2.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    #Create data table
    df = pd.DataFrame(np.vstack([colNames, perf_metrics]))
    #Plot table
    ax.table(cellText=df.values, loc='center')
    #Fix layout
    fig2.tight_layout()

    #Show table
    plt.show()  
    fig2.savefig("testing/exhaustive_testing/performance_metrics.png")

    #Save the SVM to an external data file
    sigSVM.save_svm()
    
    ##########################################################################
    #Exhaustive Testing, Cross validation, for train size 80% and 120 feature clusters
    iterations = 20
    #List to store the scores
    scores = []
    #Copy of features and target
    X_cross = X.copy()
    y_cross = y.copy()
    
    for i in range(iterations):
        
        #Shuffle the data
        X_cross, y_cross = shuffle(X_cross,y_cross, random_state=randomCharSeed)
        #Create signature SVM, no output is need at this point
        sigSVM = signature_svm(n_clusters=120, write_output=False)
                
        #Split the data into train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X_cross, y_cross, train_size=size, random_state=randomCharSeed)
        
        #Fit the model
        sigSVM.fit(X_train, y_train)
        
        #Append score
        scores.append(sigSVM.score(X_test, y_test))
    
    #Save mean score
    mean_score = np.mean(scores)
    
    #Create figure
    fig3 = plt.figure(num=3, figsize=(10,8)) 
    x_axes = range(1, iterations + 1)
    #Plot the test scores for the different iterations
    #Create subplot
    ax = fig3.add_subplot(111)
    #Draw test scores
    ax.plot(x_axes, scores, marker = 'o', label = "Scores")
    #Draw mean score
    ax.plot(x_axes, np.repeat(mean_score, iterations), label = "Mean score")
    #Add legend
    ax.legend(loc="lower right")
    #Add Title
    ax.set_title("K-fold cross validation")
    #Add x-axis label
    ax.set_xlabel("Iteration")
    #Add y-axis label
    ax.set_ylabel("Mean score")
    plt.show()
    fig3.savefig("testing/exhaustive_testing/k_fold_cross_valid.png")
    

    ##########################################################################
    #Exhaustive Testing, Train size and cluster size are variable
    train_size = np.arange(0.1, 1, 0.1)
    cluster_size = np.arange(20, 130, 20)
    #Copy data for training and testing
    X_ex = X.copy()
    y_ex = y.copy()
    
    #Overall scores
    scores_ex = []
    for csize in cluster_size:
        #Scores for actual size
        score_act_size = []
        for tsize in train_size:
            #Shuffle the data
            X_ex, y_ex = shuffle(X_ex,y_ex, random_state=randomCharSeed)
            #Create signature SVM, no output is need at this point
            sigSVM = signature_svm(n_clusters=csize, write_output=False)
                    
            #Split the data into train and test data
            X_train, X_test, y_train, y_test = train_test_split(
                X_ex, y_ex, train_size=tsize, random_state=randomCharSeed)
            
            #Fit the model
            sigSVM.fit(X_train, y_train)
            
            #Append score
            score_act_size.append(sigSVM.score(X_test, y_test))
        #Append scores for actual cluster size    
        scores_ex.append(score_act_size)
    
    #Create figure
    fig4 = plt.figure(num=4, figsize=(16,16)) 
    x_axis = train_size.copy()
    y_axis = np.array(scores_ex)
    #Subplot counter
    i = 0
    
    #Plot the results for every clustersize
    for csize in cluster_size:
        plot_act_score(x_axis, y_axis[i], csize, 230,  i + 1)
        i += 1    
    plt.show()
    
    fig4.savefig("testing/exhaustive_testing/param_search.png")
    
    ##########################################################################
    #Test if no thinning leads to higher accuracy
    
    #Training once and simple testing without thinning
    #Seed for random number generator
    randomCharSeed = 80827983   #or randomCharSeed = None
    #Train size
    size = 0.8
    
    #Create signature SVM with 120 clusters for feature clustering and a given random seed
    sigSVM_no_thin = signature_svm(n_clusters=120, random_state = randomCharSeed, write_output=False)
    
    #Get the feature matrix and the target (labels) of all images in the signature folder
    X_no_thin, y_no_thin = sigSVM_no_thin.get_img_library_Data(thin=False)
    
    #Split the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_no_thin, y_no_thin, train_size=size, random_state=randomCharSeed)
    
    #Fit the model
    sigSVM_no_thin.fit(X_train, y_train)
    
    #Compute score
    score = sigSVM_no_thin.score(X_test, y_test)
    
    #predict classes of all images in the image library
    y_hat = sigSVM_no_thin.predict(X_no_thin)
    
    #Get classes
    clf_classes = np.array(sigSVM_no_thin.get_clf_obj().classes_, int)
    #Create the confusion matrix from the result
    conf_matrix = confusion_matrix(y_no_thin, y_hat, labels = clf_classes)

    #Create the confusion matrix
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=clf_classes)
    #Display the confusion matrix
    disp.plot()
    
    #save plot
    disp.figure_.savefig("testing/exhaustive_testing/confusion_matrix_no_thin.png")

    #Get the entries of the confusion matrix, where
    #TP...True positives: Valid pictures classified as valid
    #TN...False positives: Valid pictures classified as invalid
    #FN...False negatives: Invalid pictures classified as valid
    #FP...False positives: Invalid pictures classified as invalid
    (TP, FP, FN, TN) = np.reshape(conf_matrix, 4)
    
    #Get performance metrics and print them 
    accuracy = (TP + TN) / (TP + TN + FN + FN)
    print("Accuracy: ", accuracy)
    specifity = TN / (TN + FP) 
    print("specifity: ", specifity)
    precision = TP / (TP + FP)
    print("precision: ", precision)
    recall = TP / (TP + FN)
    print("recall: ", recall)

    #Store metrics
    perf_metrics = [accuracy, specifity, precision, recall]
    #Create column names for a table
    colNames = ["accuracy", "specifity", "precision", "recall"]
    
    #Create subplot
    fig6, ax = plt.subplots()

    # hide axes
    fig6.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    #Create data table
    df = pd.DataFrame(np.vstack([colNames, perf_metrics]))
    #Plot table
    ax.table(cellText=df.values, loc='center')
    #Fix layout
    fig6.tight_layout()

    #Show table
    plt.show()  
    fig6.savefig("testing/exhaustive_testing/performance_metrics_no_thin.png")
    
    ##########################################################################
    #Exhaustive Testing, Cross validation, for train size 80% and 120 feature clusters
    iterations = 20
    #List to store the scores
    scores = []
    #Copy of features and target
    X_cross = X_no_thin.copy()
    y_cross = y_no_thin.copy()
    
    for i in range(iterations):
        
        #Shuffle the data
        X_cross, y_cross = shuffle(X_cross,y_cross, random_state=randomCharSeed)
        #Create signature SVM, no output is need at this point
        sigSVM = signature_svm(n_clusters=120, write_output=False)
                
        #Split the data into train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X_cross, y_cross, train_size=size, random_state=randomCharSeed)
        
        #Fit the model
        sigSVM.fit(X_train, y_train)
        
        #Append score
        scores.append(sigSVM.score(X_test, y_test))
    
    #Save mean score
    mean_score = np.mean(scores)
    
    #Create figure
    fig7 = plt.figure(num=7, figsize=(10,8)) 
    x_axes = range(1, iterations + 1)
    #Plot the test scores for the different iterations
    #Create subplot
    ax = fig3.add_subplot(111)
    #Draw test scores
    ax.plot(x_axes, scores, marker = 'o', label = "Scores")
    #Draw mean score
    ax.plot(x_axes, np.repeat(mean_score, iterations), label = "Mean score")
    #Add legend
    ax.legend(loc="lower right")
    #Add Title
    ax.set_title("K-fold cross validation")
    #Add x-axis label
    ax.set_xlabel("Iteration")
    #Add y-axis label
    ax.set_ylabel("Mean score")
    plt.show()
    fig3.savefig("testing/exhaustive_testing/k_fold_cross_valid_no_thin.png")


    


    
