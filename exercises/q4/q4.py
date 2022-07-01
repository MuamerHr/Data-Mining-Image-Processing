# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""
#import necessary libraries
import cv2 as cv
import matplotlib.pyplot as plt
from enhanceImage import getKernel, enhanceSharpnes

#read image
img = cv.imread("hw3_road_sign_school_blurry.jpg", cv.IMREAD_GRAYSCALE)

#show original image
plt.imshow(img, cmap='gray')

#kernel sizes
s1 = 3
s2 = 5

#rectangular kernel, 3x3
ker_rect3 = getKernel('rect', ksize_x=s1, ksize_y=s1)

#rectangular kernel, 5x5
ker_rect5 = getKernel('rect', ksize_x=s2, ksize_y=s2)

#elliptic kernel, 3x3
ker_ell3 = getKernel('ellipse', ksize_x=s1, ksize_y=s1)

#elliptic kernel, 5x5
ker_ell5 = getKernel('ellipse', ksize_x=s2, ksize_y=s2)

#cross kernel, 3x3
ker_cross3 = getKernel('cross', ksize_x=s1, ksize_y=s1)

#cross kernel, 5x5
ker_cross5 = getKernel('cross', ksize_x=s2, ksize_y=s2)

#process the original image and get the enhanced image
#enhanced image, rectangular kernel, 3x3
enhanced_1 = enhanceSharpnes(img, ker_rect3, nrIter=10)
#enhanced image, rectangular kernel, 5x5
enhanced_2 = enhanceSharpnes(img, ker_rect5, nrIter=10)
#enhanced image, elliptical kernel, 3x3
enhanced_3 = enhanceSharpnes(img, ker_ell3, nrIter=10)
#enhanced image, elliptical kernel, 5x5
enhanced_4 = enhanceSharpnes(img, ker_ell5, nrIter=10)
#enhanced image, cross kernel, 3x3
enhanced_5 = enhanceSharpnes(img, ker_cross3, nrIter=10)
#enhanced image, cross kernel, 5x5
enhanced_6 = enhanceSharpnes(img, ker_cross5, nrIter=10)

#save images
cv.imwrite('out/enhanced_rec_33.jpg', enhanced_1)
cv.imwrite('out/enhanced_rec_55.jpg', enhanced_2)
cv.imwrite('out/enhanced_ell_33.jpg', enhanced_3)
cv.imwrite('out/enhanced_ell_55.jpg', enhanced_4)
cv.imwrite('out/enhanced_cross_33.jpg', enhanced_5)
cv.imwrite('out/enhanced_cross_55.jpg', enhanced_6)

#ploting results
fig1 =plt.figure(num=1, figsize=(12, 12))
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title("original image")
plt.subplot(222), plt.imshow(enhanced_2, cmap='gray')
plt.title("enh. image, rect. kern., 5x5")
plt.subplot(223), plt.imshow(enhanced_4, cmap='gray')
plt.title("enh. image, ell. kern., 5x5")
plt.subplot(224), plt.imshow(enhanced_6, cmap='gray')
plt.title("enh. image, cross. kern., 5x5")
fig1.savefig("out/enh_together.jpg")

#ploting results
fig1 =plt.figure(num=1, figsize=(12, 12))
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title("original image")
plt.subplot(222), plt.imshow(enhanced_1, cmap='gray')
plt.title("enh. image, rect. kern., 3x3")
plt.subplot(223), plt.imshow(enhanced_3, cmap='gray')
plt.title("enh. image, ell. kern., 3x3")
plt.subplot(224), plt.imshow(enhanced_4, cmap='gray')
plt.title("enh. image, cross. kern., 3x3")
fig1.savefig("out/enh_33_together.jpg")

#Conclusion:
#Depending on the used kernel and its size but also
#depending on the forms in the original picture (circles, horizontals, diagonals)
#the algorithm sharpens certain parts of the original picture and other parts
#get getting thined out (compare rect. kernel or cross kernel 5x5, holes on heads).

#############################################################################################
#Another try with different number of dilation and eroding iterations, but with fewer overall
#iterations!

#rectangular kernel,
ker_rect = getKernel('rect', ksize_x=s1, ksize_y=s1)

#elliptic kernel,
ker_ell = getKernel('ellipse', ksize_x=s1, ksize_y=s1)

#cross kernel,
ker_cross = getKernel('cross', ksize_x=s1, ksize_y=s1)

#Process the original image and get the enhanced image
#enhanced image, rectangular kernel
enhanced_rect = enhanceSharpnes(img, ker_rect, nrIter=3, dilIter=3, erodIter=2)
#enhanced image, rectangular kernel
enhanced_ell = enhanceSharpnes(img, ker_ell, nrIter=3, dilIter=3, erodIter=2)
#enhanced image, elliptical kernel
enhanced_cross = enhanceSharpnes(
    img, ker_cross, nrIter=3, dilIter=3, erodIter=2)

#save images
cv.imwrite('out/enhanced_rec_33_5iter.jpg', enhanced_rect)
cv.imwrite('out/enhanced_ell_33_5iter.jpg', enhanced_ell)
cv.imwrite('out/enhanced_cross_33_5iter.jpg', enhanced_cross)

#ploting results
fig2 = plt.figure(num=2, figsize=(14, 7))
plt.subplot(141), plt.imshow(img, cmap='gray')
plt.title("original image")
plt.subplot(142), plt.imshow(enhanced_rect, cmap='gray')
plt.title("enh. image, rect. kern., 3x3")
plt.subplot(143), plt.imshow(enhanced_ell, cmap='gray')
plt.title("enh. image, ell. kern., 3x3")
plt.subplot(144), plt.imshow(enhanced_cross, cmap='gray')
plt.title("enh. image, cross. kern., 3x3")
fig2.savefig("out/enh_iter_together.jpg")

#Conclusion:
#The last result shows, the algorithm takes less time but performs
#better in sharpening the picture.

########################################################
#Another approach
#Lets take a threshold image of the original image
_, threshold = cv.threshold(img, 69, 255, 0)

#Take the bitwise AND of the original image and the threshold picture
img_and_thres = cv.bitwise_and(img, threshold)
img_and_thres =cv.medianBlur(img_and_thres, 3)

#This approach also enhances certain regions good.
fig3 = plt.figure(num=3, figsize=(14, 14))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title("original image")
plt.subplot(132), plt.imshow(threshold, cmap='gray')
plt.title("Threshold image")
plt.subplot(133), plt.imshow(img_and_thres, cmap='gray')
plt.title("Threshold OTSU and bit-wise AND with original")
fig3.savefig("out/thres_approach.jpg")


#Process the original threshold image and get the enhanced image
#enhanced image, rectangular kernel
enhanced_rect = enhanceSharpnes(img_and_thres, ker_rect, nrIter=3, dilIter=1, erodIter=1)
#enhanced image, rectangular kernel
enhanced_ell = enhanceSharpnes(img_and_thres, ker_ell, nrIter=3, dilIter=1, erodIter=1)
#enhanced image, elliptical kernel
enhanced_cross = enhanceSharpnes(img_and_thres, ker_cross, nrIter=3, dilIter=1, erodIter=1)

#This approach also enhances certain regions good.
fig4 = plt.figure(num=4, figsize=(14, 14))
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title("original image")
plt.subplot(222), plt.imshow(enhanced_rect, cmap='gray')
plt.title("Enhanced image, rect kernel")
plt.subplot(223), plt.imshow(enhanced_ell, cmap='gray')
plt.title("Enhanced image, rect kernel")
plt.subplot(224), plt.imshow(enhanced_cross, cmap='gray')
plt.title("Enhanced image, rect kernel")
fig4.savefig("out/enh_thres_approach.jpg")