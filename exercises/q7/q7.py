# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""

import cv2 as cv
import matplotlib.pyplot as plt
from puzzleAssembly import assemblePuzzle, SiftParams


#Read images
img_puzz = cv.cvtColor(cv.imread("hw5_puzzle_pieces.jpg"),cv.COLOR_BGR2RGB)
img_ref = cv.cvtColor(cv.imread("hw5_puzzle_reference.jpg"),cv.COLOR_BGR2RGB)

#First we perform feature matching on whole reference image
#Different SIFT parameters from bad to very good
#params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 9, nOctLayRef = 6, contrThrPuzz = 0.01, contrThrRef = 0.01, edgeThrPuzz = 26, edgeThrRef = 31, sigPuzz = 1, sigRef = 1)
#params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 10, nOctLayRef = 10, contrThrPuzz = 0.0001, contrThrRef = 0.0001, edgeThrPuzz = 980, edgeThrRef = 1010, sigPuzz = 0.9, sigRef = 0.9)
#params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 11, nOctLayRef = 9, contrThrPuzz = 0.0001, contrThrRef = 0.0001, edgeThrPuzz = 980, edgeThrRef = 1010, sigPuzz = 0.9, sigRef = 0.9)
#params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 11, nOctLayRef = 9, contrThrPuzz = 0.0001, contrThrRef = 0.0001, edgeThrPuzz = 985, edgeThrRef = 1010, sigPuzz = 0.9, sigRef = 0.9)
#params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 12, nOctLayRef = 10, contrThrPuzz = 0.0001, contrThrRef = 0.0001, edgeThrPuzz = 9890, edgeThrRef = 10160, sigPuzz = 0.9, sigRef = 0.9)
#params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 12, nOctLayRef = 9, contrThrPuzz = 0.00014, contrThrRef = 0.0001, edgeThrPuzz = 9890, edgeThrRef = 10165, sigPuzz = 0.9, sigRef = 0.9)
#params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 14, nOctLayRef = 7, contrThrPuzz = 0.00014, contrThrRef = 0.0001, edgeThrPuzz = 9896, edgeThrRef = 10165, sigPuzz = 0.9, sigRef = 0.9)
#params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 13, nOctLayRef = 7, contrThrPuzz = 0.00014, contrThrRef = 0.0001, edgeThrPuzz = 9895, edgeThrRef = 10165, sigPuzz = 0.9, sigRef = 0.9)

#Empirically best parameters
params = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 15, nOctLayRef = 7, contrThrPuzz = 0.00014, contrThrRef = 0.0001, edgeThrPuzz = 8900, edgeThrRef = 10165, sigPuzz = 0.95, sigRef = 0.9)
      
#Get assembled puzzle and not matched pieces
notMatched, puzzle = assemblePuzzle(img_puzz, img_ref, params)

#Apply median filter to filter noise on the puzzle edges.
puzzle = cv.medianBlur(puzzle, ksize = 3)

#Conclusion:
#The results really depend on the choice of SIFT parameters. Several test showed
#that the most important parameters are the number of octave layers, the
#contrast threshold and the edge threshold. The pieces need a lot more octave layers
#then the reference picture and also the edge threshold has to be smaller.

############################################################################################
#now we try the region search

#Parameter for ROI feature matching
params_roi = SiftParams(nFeatPuzz = 0, nFeatRef = 0, nOctLayPuzz = 16, nOctLayRef = 12, contrThrPuzz = 0.0001, contrThrRef = 0.0001, edgeThrPuzz = 7500, edgeThrRef = 10050, sigPuzz = 0.9, sigRef = 0.9)
  
#Get assembled puzzle and not matched pieces
notMatched_roi, puzzle_roi = assemblePuzzle(img_puzz, img_ref, params_roi, searchOnRegions = True, ybord=390, xbord = 195)

#Apply median filter to filter noise on the puzzle edges.
puzzle_roi = cv.medianBlur(puzzle_roi, ksize = 3)

#Conclusion:
#In this case results not only depend on the choice of SIFT parameters but also of ROI window.
#Sometimes smaller windows perform well for some pieces, but perform very bad for others.
#It can happen that, no homography can be found at all! In genereal bigger ROI windows performed
#better.

############################################################################################
#Show results side by side
fig = plt.figure(num=1, figsize=(14,8))
plt.subplot(131),plt.imshow(img_ref),plt.title("reference image")
plt.subplot(132),plt.imshow(puzzle),plt.title("assembled puzzle, variant whole image")
plt.subplot(133),plt.imshow(puzzle_roi),plt.title("assembled puzzle, variant ROI")

#Save result
fig.savefig('out/result.jpg')



