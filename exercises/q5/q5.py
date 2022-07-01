# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""
# import necessary libraries
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from utils import getEdges, getRotation

# read original image
img = cv.imread("hw5_insurance_form.jpg")

# save image dimensions
(h, w) = img.shape[:2]

# convert to grayscale image for edge detection
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# save grayscale image
cv.imwrite('out/1_grayscale.jpg', gray)
###################################################################################################################

# Perform canny edge detection to indicate existing edges
thr1 = 50
thr2 = 200

edges = cv.Canny(gray, thr1, thr2)

# save image of found edges
cv.imwrite('out/2_canny_first.jpg', edges)

###################################################################################################################

# find lines with hough transformation in the canny edge detection image
lines = getEdges(edges, rho=0.5, theta=1 * np.pi / 180.0,
                 threshold=180, minLineLength=20, maxGap=100)
imgWithLines = img.copy()

for line in lines:
    # extract end points of lines
    x1, y1, x2, y2 = line[0]
    # and draw them to original image
    cv.line(imgWithLines, (x1, y1), (x2, y2), (255, 0, 0), 1)

# save image of hough trans
cv.imwrite('out/3_hough_first.jpg', imgWithLines)
###################################################################################################################

# determine rotation of image with help of found lines
# save coordinates of first found line
(x1, y1, x2, y2) = lines[0, 0]

# get rotation angle in rad and convert to deg
rho = getRotation(x1, y1, x2, y2) * 180 / np.pi

# get rotation matrix with rotation angle rho
rotMatrix = cv.getRotationMatrix2D((int(x1), int(y1)), rho, 1.0)

# rotate the picture and assign generated blank space white color
rotated = cv.warpAffine(img, rotMatrix, (int(w), int(h)),
                        borderValue=(255, 255, 255))

# save image of rotated img
cv.imwrite('out/4_rotated.jpg', rotated)

###################################################################################################################
# perform edge detection once again on rotated picture
# and draw the lines to fill gaps. One could draw the lines into
# the original picture, but not every line was found and
# the resulting image has aliasing effects (stair effects on lines).
# Therefore we perform the edge detection once again.

imgWithLines_rot = rotated.copy()

# Perform canny edge detection to indicate existing edges
thr1 = 50
thr2 = 200

edges_rot = cv.Canny(imgWithLines_rot, thr1, thr2)
###################################################################################################################

# find lines with hough transformation in the canny edge detection image
# first the vertical lines and the most of the horizontal
lines_vert = getEdges(edges_rot, rho=1, theta=1 * np.pi /
                      180.0, threshold=300, minLineLength=100, maxGap=100)
# then the remaining horizontal lines
lines_hor = getEdges(edges_rot, rho=1, theta=17 * np.pi /
                     180.0, threshold=200, minLineLength=20, maxGap=150)
for line in lines_vert:
    # extract end points of lines
    x1, y1, x2, y2 = line[0]
    # and draw them to original image
    cv.line(imgWithLines_rot, (x1, y1), (x2, y2), (255, 0, 0), 4)

# save image of the first hough trans
cv.imwrite('out/5_hough_vert_hor.jpg', imgWithLines_rot)

for line in lines_hor:
    # extract end points of lines
    x1, y1, x2, y2 = line[0]
    # and draw them to original image
    cv.line(imgWithLines_rot, (x1, y1), (x2, y2), (0, 255, 0), 4)

# save image of the second hough trans
cv.imwrite('out/6_hough_hor_rem.jpg', imgWithLines_rot)


###################################################################################################################
# process the endresult
img_out = rotated.copy()

for line in lines_vert:
    # extract end points of lines
    x1, y1, x2, y2 = line[0]
    # and draw them to original image
    cv.line(img_out, (x1, y1), (x2, y2), (0, 0, 0), 1)

for line in lines_hor:
    # extract end points of lines
    x1, y1, x2, y2 = line[0]
    # and draw them to original image
    cv.line(img_out, (x1, y1), (x2, y2), (0, 0, 0), 1)

# 1. Close the image
# define kernel for morphological operations
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

# dilate the image with the given kernel and save it
img_e = cv.erode(img_out, kernel, iterations=1)

# erode the image with the given kernel and save it
img_d = cv.dilate(img_e, kernel, iterations=1)

# save image morphological operation closing
cv.imwrite('out/7_moprh_closing.jpg', img_d)

# 2. Sharpen the image
# define sharpening kernel
kernel = -np.ones((3, 3), np.float32)/9
kernel[1, 1] += 2

# sharpen the picture twice
img_out = cv.filter2D(img_d, -1, kernel)
img_out = cv.filter2D(img_d, -1, kernel)

# show results
plt.figure(num=1, figsize=(14, 8))
plt.subplot(121), plt.imshow(img), plt.title("original image")
plt.subplot(122), plt.imshow(img_d), plt.title("process image")

# save result
cv.imwrite('out/8_result.jpg', img_out)

