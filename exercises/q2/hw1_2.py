# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the images as grayscale
img_1  = cv2.imread("hw1_dark_road_1.jpg",0)
img_2  = cv2.imread("hw1_dark_road_2.jpg",0)
img_3  = cv2.imread("hw1_dark_road_3.jpg",0)
#calc the histograms
hist_1 = cv2.calcHist([img_1],[0],None,[256],[0,256])
hist_2 = cv2.calcHist([img_2],[0],None,[256],[0,256])
hist_3 = cv2.calcHist([img_3],[0],None,[256],[0,256])

# show imgs+histograms
fig=plt.figure("Histograms")
plt.subplot(3,2,1), plt.imshow(img_1, 'gray')
plt.subplot(3,2,2), plt.plot(hist_1)
plt.subplot(3,2,3), plt.imshow(img_2, 'gray')
plt.subplot(3,2,4), plt.plot(hist_2)
plt.subplot(3,2,5), plt.imshow(img_3, 'gray')
plt.subplot(3,2,6), plt.plot(hist_3)
# save the img+histograms
fig.savefig("out/histograms.jpg")
"""
Image 1 has very low contrast and this can be seen by most of the intensity values 
ranging from 0 to around 40
The same applies for image 2 and 3 although for image 3 there are less pixels with 
exacly the same intensity and the range of the values is not as narrow as for the 
other images     
"""
# use inbuilt fct to globally eq. histograms
equ_1 = cv2.equalizeHist(img_1)
equ_2 = cv2.equalizeHist(img_2)
equ_3 = cv2.equalizeHist(img_3)
# calc histograms
hist_1 = cv2.calcHist([equ_1],[0],None,[256],[0,256])
hist_2 = cv2.calcHist([equ_2],[0],None,[256],[0,256])
hist_3 = cv2.calcHist([equ_3],[0],None,[256],[0,256])
# show results
fig2=plt.figure("Globally equalized Histograms")
plt.subplot(3,2,1), plt.imshow(equ_1, 'gray')
plt.subplot(3,2,2), plt.plot(hist_1)
plt.subplot(3,2,3), plt.imshow(equ_2, 'gray')
plt.subplot(3,2,4), plt.plot(hist_2)
plt.subplot(3,2,5), plt.imshow(equ_3, 'gray')
plt.subplot(3,2,6), plt.plot(hist_3)
# save the globally eq. histograms
fig2.savefig("out/global_equalized_histograms.jpg")
"""
For all three images the regions  where there are problems is the sky, because 
by equalizing the histograms we introduced a lot of noise there
For the other regions in the image this procedure produced quite nice results with 
way more of the image beeing visible after the equalization
"""

# in our opinion best settings for the contrast limited adaptive hist. eq.
clahe1 = cv2.createCLAHE(clipLimit=7.1, tileGridSize=(12,12))
# apply to all images
bl1 = clahe1.apply(img_1)
bl2 = clahe1.apply(img_2)
bl3 = clahe1.apply(img_3)
# calculate the histograms
hist_1 = cv2.calcHist([bl1],[0],None,[256],[0,256])
hist_2 = cv2.calcHist([bl2],[0],None,[256],[0,256])
hist_3 = cv2.calcHist([bl3],[0],None,[256],[0,256])
#show results
fig3 = plt.figure("Adaptive equalized Histograms")
plt.subplot(3,2,1), plt.imshow(bl1, 'gray')
plt.subplot(3,2,2), plt.plot(hist_1)
plt.subplot(3,2,3), plt.imshow(bl2, 'gray')
plt.subplot(3,2,4), plt.plot(hist_2)
plt.subplot(3,2,5), plt.imshow(bl3, 'gray')
plt.subplot(3,2,6), plt.plot(hist_3)
# write the figure into an image
fig3.savefig("out/adaptive_equalized_histograms.jpg")
"""
For the best results a cliLimit between 7 and 8 would be optimal
a value closer to 7 results in less noise, but a darker image in general
a value closer to 8 produces more noise but increases the brightness 
We think it is more important to keep the noise to a minimum, while inreasing the quality
of the image --> 7.1
A gridsize of 144 rectangular tiles  worked pretty well,
so we diced to not go with the default of 64 tiles, since the 12x12 version 
produces results with more details

"""
# show comparison
fig4 = plt.figure("Original - Global - Adaptive")
plt.subplot(3,3,1), plt.imshow(img_1, 'gray'), plt.title("original")
plt.subplot(3,3,2), plt.imshow(equ_1, 'gray'), plt.title("global eq.")
plt.subplot(3,3,3), plt.imshow(bl1, 'gray'), plt.title("locally adaptive eq.")
plt.subplot(3,3,4), plt.imshow(img_2, 'gray')
plt.subplot(3,3,5), plt.imshow(equ_2, 'gray')
plt.subplot(3,3,6), plt.imshow(bl2, 'gray')
plt.subplot(3,3,7), plt.imshow(img_3, 'gray')
plt.subplot(3,3,8), plt.imshow(equ_3, 'gray')
plt.subplot(3,3,9), plt.imshow(bl3, 'gray')
fig4.savefig("out/comparison.jpg")
"""
The quality of the image produced with the adaptive histogram eq. is 
way better (subjectively) compared to the resuls achieved with 
the global histogram equalization
The noise is mostly removed while still mainting a way better contrast than 
in the original image
"""