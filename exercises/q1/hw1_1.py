# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""

import numpy as np
import cv2 
import matplotlib.pyplot as plt

# load images and make gray copies of it for display purposes
input_img_1 = cv2.imread("hw1_painting_2_reference.jpg")
input_img_2 = cv2.imread("hw1_painting_2_tampered.jpg")
input_img_3 = cv2.imread("hw1_painting_1_reference.jpg")
input_img_4 = cv2.imread("hw1_painting_1_tampered.jpg")
original_painting = cv2.cvtColor(input_img_1,cv2.COLOR_BGR2GRAY)
tampered_painting = cv2.cvtColor(input_img_2,cv2.COLOR_BGR2GRAY)
original_painting_2 = cv2.cvtColor(input_img_3,cv2.COLOR_BGR2GRAY)
tampered_painting_2 = cv2.cvtColor(input_img_4,cv2.COLOR_BGR2GRAY)
#counter for img labeling
counter=1


def getDifferences(original_img, tampered_img):
    """
    Parameters
    ----------
    original_img : expects a grayscale img, the original image 
    tampered_img : expects a grayscale img, the tampered image, that shall be 
                   compared to the orignal image

    Returns
    -------
    output : returns binary image with 0 = same as original_img, 
             255 = different to orignal_img

    """
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(original_img,None)
    kp2, des2 = orb.detectAndCompute(tampered_img,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # only take 10 best matches
    matches= matches[:10]
    # Draw first 10 matches.
    img3 = cv2.drawMatches(original_img,kp1,tampered_img,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # show matches and lines between them 
    plt.figure("Features_" + str(counter),figsize=(14,8)), plt.imshow(img3, "gray")
    cv2.imwrite("out/features_"+ str(counter)+".jpg",img3)
    # store coordinates of matches 
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    # get the difference in position of the matches of the keypoints
    diff = pts1 -pts2
    # get the median values to remove noise
    (delta_y,delta_x) = (np.median(diff[:,0]),np.median(diff[:,1]))
    # translation matrix that takes overlaps the images for comparison
    M = np.float32([[1,0,-delta_x],[0,1,-delta_y]])
    # perform the translation
    shifted_2=cv2.warpAffine(tampered_img, M, (tampered_img.shape[1],tampered_img.shape[0]))
    # substract the alligned tampered img from the original
    output = np.abs(np.float32(original_img) -np.float32(shifted_2))
    # thresholding to reduce noise 
    output[output>2] = 255
    output[output<=2] = 0
    
    cv2.imwrite("out/differences_"+ str(counter)+".jpg",output)
    return output


output_1=getDifferences(original_painting_2, tampered_painting_2)
# for img labeling
counter+=1
output_2=getDifferences(original_painting, tampered_painting)
# Show results
plt.figure("Tampering Detection_1",figsize=(14,8))
plt.subplot(1,3,1), plt.imshow(cv2.cvtColor( input_img_3, cv2.COLOR_BGR2RGB)), plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(cv2.cvtColor( input_img_4, cv2.COLOR_BGR2RGB)), plt.title("Tampered"), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(output_1.astype(np.uint8),'gray'), plt.title("Differences"), plt.xticks([]), plt.yticks([])

plt.figure("Tampering Detection_2",figsize=(14,8))
plt.subplot(1,3,1), plt.imshow(cv2.cvtColor( input_img_1, cv2.COLOR_BGR2RGB)), plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(cv2.cvtColor( input_img_2, cv2.COLOR_BGR2RGB)), plt.title("Tampered"), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(output_2.astype(np.uint8),'gray'), plt.title("Differences"), plt.xticks([]), plt.yticks([])