# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics

# Load images
img_1  = cv2.imread("hw3_building.jpg",0)
img_2  = cv2.imread("hw3_train.jpg",0)

# Apply median filter with 3x3 and 5x5 kernel to img1
img_1_m3 = cv2.medianBlur(img_1,3)
img_1_m5 = cv2.medianBlur(img_1,5)
# Apply median filter with 3x3 and 5x5 kernel to img2
img_2_m3 = cv2.medianBlur(img_2,3)
img_2_m5 = cv2.medianBlur(img_2,5)

# show the images
plt.figure("Median Filter")
plt.subplot(2,3,1), plt.imshow(img_1, 'gray'), plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2), plt.imshow(img_1_m3,'gray'), plt.title("3x3 Median Filter"),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3), plt.imshow(img_1_m5, 'gray'),plt.title("5x5 Median Filter"),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4), plt.imshow(img_2,'gray'),plt.title("Original"),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5), plt.imshow(img_2_m3, 'gray'),plt.title("3x3 Median Filter"),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6), plt.imshow(img_2_m5,'gray'),plt.title("5x5 Median Filter"),plt.xticks([]), plt.yticks([])

"""
For the 3x3 median filter the noise is still in the image although very reduced, while 
only slightly reducing the quality of the edges and the features the image contains
For the 5x5 median filter however the noise is practically removed, but the quality of 
the edges and the features of the images is also reduced 
This kind of tradeoff is common and it depends on what the resulting image is used for 
to decide which kind of filtering would be more adequate 
"""
# kernel for weighted median filtering
kernel = np.array([[0,1,1,1,0],[1,2,2,2,1],[1,2,4,2,1],[1,2,2,2,1],[0,1,1,1,0]])

def calcCustomMedian(img,kernel):
    # get amount of padding needed, assuming kernel = mxn matrix m=n, m odd
    offset = np.size(kernel,0)//2
    # add padding to the image for the filtering
    img_padded = cv2.copyMakeBorder(img, offset, offset, offset, offset,cv2.BORDER_REPLICATE)
    # get height and width 
    height,width = img.shape
    # copy img for output 
    new_image = img.copy()
    
    # y between the two paddings
    for y in range(offset,height+offset):
        # x between the padding and from right to left 
        for x in range(width+offset-1,offset-1,-1):
            # array for values we calculate the median from 
            values = []
            # go through every value of the kernel
            for a in range(0,5):           
                for b in range(0,5):
                    # current weight
                    w = kernel[a,b]
                    # current value
                    c_value = img_padded[y+a-offset,x+b-offset]
                    # add w amount of c_values to the list
                    for c in range(0,w):                   
                        values.append(c_value)
                        # get the median 
                        new_image[y-offset,x-offset] =statistics.median(values)
    return new_image
# calc weighted median for the 2 images
output_1 = calcCustomMedian(img_1, kernel)
output_2 = calcCustomMedian(img_2, kernel)
# show the images
plt.figure("Comparison")
plt.subplot(2,3,1), plt.imshow(output_1, 'gray'), plt.title("Weighted Median Filter"), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2), plt.imshow(img_1_m3,'gray'), plt.title("3x3 Median Filter"),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3), plt.imshow(img_1_m5, 'gray'),plt.title("5x5 Median Filter"),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4), plt.imshow(output_2,'gray'),plt.title("Weighted Median Filter"),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5), plt.imshow(img_2_m3, 'gray'),plt.title("3x3 Median Filter"),plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6), plt.imshow(img_2_m5,'gray'),plt.title("5x5 Median Filter"),plt.xticks([]), plt.yticks([])
# write the images
cv2.imwrite("out/Weighted_median_filter_building.jpg", output_1)
cv2.imwrite("out/3x3_median_filter_building.jpg", img_1_m3)
cv2.imwrite("out/5x5_median_filter_building.jpg", img_1_m5)
cv2.imwrite("out/Weighted_median_filter_train.jpg", output_2)
cv2.imwrite("out/3x3_median_filter_train.jpg", img_2_m3)
cv2.imwrite("out/5x5_median_filter_train.jpg", img_2_m5)


