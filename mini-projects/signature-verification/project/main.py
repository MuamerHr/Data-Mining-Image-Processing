# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@author:  Freislich, Hrncic, Siebenhofer

to use the code: pip install imagehash, pip install opencv-python
License: MIT
"""

import cv2 as cv
from matplotlib import pyplot as plt
from signature_svm import load_svm_from_datafile

if __name__ == "__main__":
    #Path to image
    image_path = "testing/val_f.png"
        
    #Load test image
    image = cv.imread(image_path)
    #Load trained signature_svm
    sigSVM= load_svm_from_datafile()

    #Transform image to feature vector (vocabulary + special features)
    X_img = sigSVM.image_to_feature_vector(image_path)    
    y_hat = sigSVM.predict(X_img)
    
    print("Predicted class for image:")
    print(int(y_hat))
    result = sigSVM.check_signature(image_path)

    print("Signature is " + result)

    if(result == "valid"):
        offset = (55, 35)
        value = (34, 139, 34)
    else:
        offset = (40, 35)
        value = (220, 20, 60)
    # get the result as a string
    image = cv.resize(image, (200, 100), interpolation=cv.INTER_AREA)

    color_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    color_image = cv.copyMakeBorder(
        color_image, 50, 0, 0, 0, cv.BORDER_CONSTANT, None, value)
    color_image = cv.putText(color_image, result, offset,
                             cv.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
    plt.figure("Result",figsize=(6, 6)), plt.imshow(color_image), plt.xticks([]),plt.yticks([])

