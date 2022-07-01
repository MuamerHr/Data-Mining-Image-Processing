# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""

import cv2 as cv
import numpy as np

class SiftParams:
    """
    Class which holds the attribute values relevant for the SIFT descriptors
    of the puzzle piece and the reference image.

    Attributes
    ----------
    nFeat: int
        The number of best features to retain. The features are ranked by 
        their scores (measured in SIFT algorithm as the local contrast).

    nOctLay: int
        The number of layers in each octave. 3 is the value used in D. Lowe paper. 
        The number of octaves is computed automatically from the image resolution.

    contrThr: double
        The contrast threshold used to filter out weak features in 
        semi-uniform (low-contrast) regions. The larger the threshold, 
        the less features are produced by the detector.

    edgeThr: double
        The threshold used to filter out edge-like features. Note that the 
        its meaning is different from the contrastThreshold, i.e. the larger 
        the edgeThreshold, the less features are filtered out 
        (more features are retained).

    sig: double
        The sigma of the Gaussian applied to the input image at the octave #0. 
        If your image is captured with a weak camera with soft lenses, you might 
        want to reduce the number. 
    """

    def __init__(self, nFeatPuzz=0, nFeatRef=0, nOctLayPuzz=4, nOctLayRef=4, contrThrPuzz=0.04, contrThrRef=0.04, edgeThrPuzz=10, edgeThrRef=10, sigPuzz=1.6, sigRef=1.6):
        self.nFeatPuzz = nFeatPuzz
        self.nFeatRef = nFeatRef
        self.nOctLayPuzz = nOctLayPuzz
        self.nOctLayRef = nOctLayRef
        self.contrThrPuzz = contrThrPuzz
        self.contrThrRef = contrThrRef
        self.edgeThrPuzz = edgeThrPuzz
        self.edgeThrRef = edgeThrRef
        self.sigPuzz = sigPuzz
        self.sigRef = sigRef


def saveImageOfMatchedKP(piece, image, kpPuzz, kpRef, matches, path="out/"):
    """
    This methods creates an image of the puzzle piece and the reference image
    side by side and draws the matched features into the result.
    The image is saved to a folder out/matches.

    Attributes
    ----------
    piece: Numpy array, type int32
        Image of the puzzle piece.

    image: Numpy array, type int32
        Reference image.

    index: int
        Actual piece number.

    kpPuzz: Tuple, Keypoints
        Keypoints of the puzzle.

    kpRef: Tuple, Keypoints
        Keypoints of the reference image.

    matches: List, Dmatch
        List of good matched keypoints.
    """
    # Create black image such that both images fit into it
    imageMatches = np.empty(
        (max(piece.shape[0], image.shape[0]), piece.shape[1]+image.shape[1], 3), dtype=np.uint8)
    # Draw the good matches.
    cv.drawMatches(piece, kpPuzz, image, kpRef, matches, imageMatches,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #Convert image to color image
    imageMatches = cv.cvtColor(imageMatches, cv.COLOR_BGR2RGB)

    # Save image.
    cv.imwrite(path, imageMatches)


def getBoundingBox(contour):
    """
    Method to get the bounding box of a given contour object.

    Attributes
    ----------
    contour: Numpy array, type int32
        Contour found by method cv2.findContours(...) or any other
        method for finding contours where return values are Numpy
        arrays of type Int32.

    Returns
    -------
    (x,y,w,h): int
        Origin coordinates x, y of the bounding box with the dimensions
        width and height w, h
    """
    # Get dimensions of bounding box.
    return cv.boundingRect(contour)


def getSiftKeypAndDesc(image, nFeat=0, nOctLay=3, contrThr=0.04, edgeThr=10, sig=1.6):
    """
    Method to get SIFT keypoints and descriptors.

    Attributes
    ----------
    nFeat: int
        The number of best features to retain. The features are ranked by 
        their scores (measured in SIFT algorithm as the local contrast).

    nOctLay: int
        The number of layers in each octave. 3 is the value used in D. Lowe paper. 
        The number of octaves is computed automatically from the image resolution.

    contrThr: double
        The contrast threshold used to filter out weak features in 
        semi-uniform (low-contrast) regions. The larger the threshold, 
        the less features are produced by the detector.

    edgeThr: double
        The threshold used to filter out edge-like features. Note that the 
        its meaning is different from the contrastThreshold, i.e. the larger 
        the edgeThreshold, the less features are filtered out 
        (more features are retained).

    sig: double
        The sigma of the Gaussian applied to the input image at the octave #0. 
        If your image is captured with a weak camera with soft lenses, you might 
        want to reduce the number. 

    Returns
    -------
    keypoints: Tuple, Keypoints
        Keypoints of the image.

    descriptors: Array, float32
        Descriptors of the image.

    """
    # create the sift descriptor
    sift = cv.SIFT_create(nfeatures=nFeat, nOctaveLayers=nOctLay,
                          contrastThreshold=contrThr, edgeThreshold=edgeThr, sigma=sig)

    # detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def getRegionOfInterest(image, yOrig, xOrig, regHeight, regWidth):
    """
    Method to get an image, where everything is black except a 
    rectangular region of interest (roi). 
    Since the image is reduced to the roi, feature detection 
    and comparison has much better performance.

    Attributes
    ----------
    image: Numpy array, type int32
        The Image where the region of interest is extracted.

    yOrig: int
        Y-coordinate of the rectangles origin.

    xOrig: int
        X-coordinate of the rectangles origin.

    regHeight: int
        Height of the rectangle.

    regWidth: int
        Width of the rectangle.

    Returns
    -------
    roi: Numpy array, type int32
        Image with region of interest.

    """
    # Make a copy of the original image
    roi = image.copy()
    # Make every pixel above the region of interest black
    roi[: yOrig, :] = [0, 0, 0]
    # Make every pixel under the region of interest black
    roi[yOrig + regHeight:, :] = [0, 0, 0]
    # Make every pixel left of the region of interest black
    roi[:, : xOrig] = [0, 0, 0]
    # Make every pixel right of the region of interest black
    roi[:, xOrig + regWidth:] = [0, 0, 0]

    return roi


def getContours(img, threshVal=200, maxval=255, thrType=cv.THRESH_BINARY):
    """
    Method to get the contours of the pieces.

    Attributes
    ----------
    img: Numpy array, type int32
        Image where we want to find the contours.

    threshVal: double
        Threshold value.

    maxval: double
        Maximum value to use with the THRESH_BINARY and 
        THRESH_BINARY_INV thresholding types. 

    thrType: int
        Threshold type. The following thresholds are provided:
        -cv.THRESH_BINARY
        -cv.THRESH_BINARY_INV
        -cv.THRESH_TRUNC
        -cv.THRESH_TOZERO
        -cv.THRESH_TOZERO_INV        

    Returns
    -------
    contours: List, Numpy Array, int32
        Found contour lines of the pieces.

    """
    # Convert image to grayscale
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # get edges through thresholding
    _, thresh = cv.threshold(imgray, threshVal, maxval, thrType)
    # find contours
    (contours, _) = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # sort contours by x and by y coordinate
    cont_sorted = sorted(contours, key=lambda k: (k[0][0][0], k[0][0][1]))
    # ignore the first contour, because it is the contour of the whole image
    cont_sorted = cont_sorted[1:]

    return cont_sorted


def getPiece(image, contours, i):
    """
    Method to get a single puzzle piece inside its bounding box.

    Attributes
    ----------
    img: Numpy array, type int32
        Image where we want to extract the piece.

    contour: Numpy Array, int32
        Contour line of the puzzle piece.

    Returns
    -------
    piece: Numpy Array, int32
        Puzzle piece inside its bounding box.

    """
    # Take a copy of the original image with the pieces
    imgTemp = image.copy()
    # Draw the found contours and the area enclosed
    cv.drawContours(imgTemp, contours, -1, (0, 0, 0), thickness=-1)
    # convert the image to grayscale, in order to get a threshold mask
    imgTemp = cv.cvtColor(imgTemp, cv.COLOR_BGR2GRAY)
    # Take a copy of the image
    piece = image.copy()
    # and apply the threshold mask to the image. Everything that is not
    # inside the found contours will be set to black color.
    piece[imgTemp != 0] = [0, 0, 0]

    # get the bounding box of the desired contour
    x, y, w, h = getBoundingBox(contours[i])
    # return the puzzle piece
    return piece[y:y+h, x:x+w]


def getMatches(desc_puzz, desc_ref, algTyp=1, FLANN_trees=20, FLANN_checks=150, Lowe_ratio=0.7):
    """
    Method to compute the good matches of feature detection by Lowe's ratio test.

    Attributes
    ----------
    desc_puzz: Array, float32
        Descriptors of the puzzle piece.

    desc_ref: Array, float32
        Descriptors of the reference image.

    algTyp: int
        Algorithm type for feature matching.
        Possible types:
        FLANNBASED = 1,
        BRUTEFORCE = 2,
        BRUTEFORCE_L1 = 3,
        BRUTEFORCE_HAMMING = 4,
        BRUTEFORCE_HAMMINGLUT = 5,
        BRUTEFORCE_SL2 = 6

    FLANN_trees: int
        Number of FLANN trees.

    FLANN_checks: int
        Number which specifies number of times
        the trees in the index should be recursively traversed.

    Lowe_ratio: float32
        Factor for Lowe ratio.

    Returns
    -------
    matches: List, Dmatch
        List of good matched keypoints.

    """
    # algtype:

    # Create dictionaries which specify the the FLANN algorithm.

    # A FLANN based matcher requires two dictionaries which
    # specify the algorithm to be used, its related parameters etc.

    # First one is IndexParams. For various algorithms, the information
    # to be passed is explained in FLANN docs.

    index_params = dict(algorithm=algTyp, trees=FLANN_trees)

    # Second dictionary is the SearchParams. It specifies the number of times
    # the trees in the index should be recursively traversed.
    search_params = dict(checks=FLANN_checks)

    # Create matcher.
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Try to find matches.
    knn_matches = flann.knnMatch(desc_puzz, desc_ref, k=2)

    # Save "good" matches as per Lowe's ratio test.
    matches = []
    for m, n in knn_matches:
        print(m.distance)
        if m.distance < Lowe_ratio * n.distance:
            matches.append(m)

    return matches


def findHomography(piece, image, kp_puzz, kp_ref, matches):
    """
    Method to find homography given keypoints of the puzzle piece and
    keypoints of the reference image.

    Attributes
    ----------
    piece: Numpy array, type int32
        Image of the puzzle piece.

    image: Numpy array, type int32
        Reference image.

    kpPuzz: Tuple, Keypoints
        Keypoints of the puzzle.

    kpRef: Tuple, Keypoints
        Keypoints of the reference image.

    matches: List, Dmatch
        List of good matched keypoints.

    Returns
    -------
    status: int
        Returns status = 1, if homography could be found and status = 0
        otherwise.
    fit: Numpy array, type int32
        Image of the fitted puzzle piece if homography was found and
        blank image otherwise
    """

    # Variable to indicate wether a homography could be found.
    status = 1
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp_puzz[match.queryIdx].pt
        points2[i, :] = kp_ref[match.trainIdx].pt

    #
    fit = np.zeros_like(image)

    # Find homography
    try:
        # Find homography using RANSAC
        hom, _ = cv.findHomography(points1, points2, cv.RANSAC)
        # Store reference image shape.
        (h, w) = image.shape[:2]
        # Transform image of piece with homography.
        fit = cv.warpPerspective(piece, hom, (w, h))
    except:
        # If no homography could be found (less then 4 points, and other problems).
        # status is set to zero.
        status = 0

    return status, fit


def putSinglePieceROI(piece, image, index, params: SiftParams, ybord=110, xbord=25):
    """
    Method to fit a single puzzle piece into a reference image. 
    Keypoint detection and descriptor computation is performed 
    in the reference image along a grid of regions of interest 
    of size (pieceHeight + ybord) * (pieceWidth + xbord)

    Attributes
    ----------
    piece: Numpy array, type int32
        Image of the puzzle piece.

    image: Numpy array, type int32
        Reference image.

    index: int
        Actual piece number.

    params: SiftParams
        Object which holds parameters of SIFT descriptors.

    ybord: int
        Additional height for the roi.

    xbord: int
        Additional width for the roi.    

    Returns
    -------
    status: int
        Returns status = 1, if homography could be found and status = 0
        otherwise.
    fit: Numpy array, type int32
        Image of the fitted puzzle piece if homography was found and
        blank image otherwise
    """
    # Search for keypoints and descriptos search in regions of interest
    # of the reference image.

    # Find Keypoints and compute descriptors of the piece.
    kp_puzz, desc_puzz = getSiftKeypAndDesc(
        piece, nFeat=0, nOctLay=params.nOctLayPuzz, contrThr=params.contrThrPuzz, edgeThr=params.edgeThrPuzz, sig=params.sigPuzz)

    # Status variable which indicates, if a homography could be found.
    status = 0

    # Variable of the fitted piece
    fit = np.zeros_like(image)

    # store image dimensions
    (href, wref) = image.shape[:2]
    (pieceHeight, pieceWidth) = piece.shape[:2]

    # Calculate the number of whole pieces with the desired height on the vertical axis.
    # The height is the sum of pieceheight and ybord.
    qv = int(href / (pieceHeight + ybord))

    # Calculate the number of whole pieces with the desired width on the horizontal axis.
    # The height is the sum of pieceheight and ybord.
    qh = int(wref / (pieceWidth + xbord))

    # Set region height.
    h = pieceHeight + ybord
    # Set region width.
    w = pieceWidth + xbord
    path = "out/roi/matches/matches_piece_" + str(index + 1) + ".jpg"
    # Try to fit the piece on the grid of regions.
    for i in range(qv + 1):
        for j in range(qh + 1):
            # Obtain region of interest
            # Region with the dimensions of the piece + border
            if(i < qv and j < qh):
                y = i * (pieceHeight + ybord)
                x = j * (pieceWidth + xbord)

            # Rightmost region in horizontal strip of height = pieceheight + border
            elif(i < qv and j >= qh):
                y = i * (pieceHeight + ybord)
                x = wref - (pieceWidth + xbord)

            # Bottom region in vertical strip of width = pieceWidth + border
            elif(i >= qv and j < qh):
                y = href - (pieceHeight + ybord)
                x = j * (pieceWidth + xbord)

            # Region in the right lower corner
            elif(i >= qv and j >= qh):
                y = href - (pieceHeight + ybord)
                x = wref - (pieceWidth + xbord)

            # Get Region of Interest
            roi = getRegionOfInterest(image, y, x, h, w)

            # Find keypoints and compute in region of interest.
            kp_ref, desc_ref = getSiftKeypAndDesc(
                roi, nFeat=0, nOctLay=params.nOctLayRef, contrThr=params.contrThrRef, edgeThr=params.edgeThrRef, sig=params.sigRef)

            # Execute FLANN based feature matching and save matches.
            matches = getMatches(desc_puzz, desc_ref)

            # Try to compute homography.
            status, fit = findHomography(
                piece, image, kp_puzz, kp_ref, matches)

            if(status == 1):
                # Homography could be found. Save Image.
                saveImageOfMatchedKP(piece, image,
                                     kp_puzz, kp_ref, matches, path)

                # Return status and the fitted piece.
                return status, fit

    return status, fit


def putSinglePiece(piece, image, index, params: SiftParams, kp, desc):
    """
    Method to fit a single puzzle piece into a reference image. 
    Keypoint detection and descriptor computation is performed 
    in the reference image along a grid of regions of interest 
    of size (pieceHeight + ybord) * (pieceWidth + xbord)

    Attributes
    ----------
    piece: Numpy array, type int32
        Image of the puzzle piece.

    image: Numpy array, type int32
        Reference image.

    index: int
        Actual piece number.

    params: SiftParams
        Object which holds parameters of SIFT descriptors.

    kp: Tuple, Keypoints
        Keypoints of the reference image.

    desc: Array, float32
        Descriptors of the reference image.


    Returns
    -------
    status: int
        Returns status = 1, if homography could be found and status = 0
        otherwise.
    fit: Numpy array, type int32
        Image of the fitted puzzle piece if homography was found and
        blank image otherwise
    """
    # Try to fit a single piece.
    # Search for keypoints and descriptos in the whole reference image.

    # Status variable which indicates, if a homography could be found.
    status = 0

    # Find Keypoints and compute descriptors of the piece.
    kp_puzz, desc_puzz = getSiftKeypAndDesc(
        piece, nFeat=0, nOctLay=params.nOctLayPuzz, contrThr=params.contrThrPuzz, edgeThr=params.edgeThrPuzz, sig=params.sigPuzz)

    # Variable to hold the fitted piece.
    fit = np.zeros_like(image)

    # Execute FLANN based feature matching and save matches.
    matches = getMatches(desc_puzz, desc)

    # Try to find Homography. Return status and the fitted piece.
    status, fit = findHomography(piece, image, kp_puzz, kp, matches)
    path = "out/whole_image/matches/matches_piece_" + str(index + 1) + ".jpg"
    if(status == 1):
        # Homography could be found. Save Image.
        saveImageOfMatchedKP(piece, image, kp_puzz, kp, matches, path)

    return status, fit


def assemblePuzzle(img_puzz, image, params: SiftParams, searchOnRegions=False, ybord=0, xbord=0):
    """
    Method to assemble a puzzle of pieces through feature matching.

    Attributes
    ----------
    piece: Numpy array, type int32
        Image of the puzzle piece.

    image: Numpy array, type int32
        Reference image.

    params: SiftParams
        Object which holds parameters of SIFT descriptors.

    searchOnRegions: bool
        Variable to indicate wether feature matching is performed
        on the whole reference image or on a grid of regions of
        interest. 

    Returns
    -------
    puzzle: Numpy array, type int32
        Image of assembled puzzle.
    notMatchedPieces: List, Numpy array, type int32
        List of contours/pieces which could not be fitted.
    """
    # Variable for final image
    puzzle = np.zeros_like(image)

    # Get contours of the puzzle pieces
    contours = getContours(img_puzz)

    # Variable for not matched pieces
    notMatchedPieces = []

    # Number of contours
    n = len(contours)

    # If keypoint and descriptor search is performed on the whole
    # reference image, we only have to do this once.
    if(not searchOnRegions):
        # Find keypoints and compute of the whole reference image.
        kp_ref, desc_ref = getSiftKeypAndDesc(
            image, nFeat=0, nOctLay=params.nOctLayRef, contrThr=params.contrThrRef, edgeThr=params.edgeThrRef, sig=params.sigRef)

    #outputpath for assembled pieces
    path = "out/whole_image/" if not searchOnRegions else "out/roi/"

    # Main algorithm.
    for i in range(n):

        # Get a single puzzle piece
        piece = getPiece(img_puzz, contours, i)

        if(not searchOnRegions):
            # Try to find homography and to draw piece into assembly
            status, fit = putSinglePiece(
                piece, image, i, params, kp_ref, desc_ref)
        else:
            status, fit = putSinglePieceROI(
                piece, image, i, params, ybord, xbord)

        #Convert image to color image
        fit = cv.cvtColor(fit, cv.COLOR_BGR2RGB)

        # If status equals 1, a homography was found
        if(status == 1):
            # Put the piece into the assembly
            puzzle = cv.add(puzzle, fit)

            # Write grayscale image of assembly
            cv.imwrite(str(path + "assembling/piece_" +
                       str(i + 1) + ".jpg"), puzzle)
        else:
            # mark piece i as not fitted
            notMatchedPieces.append(contours[i])

    # Print assembly status.
    if(len(notMatchedPieces) == 0):
        print("All pieces are fitted.")
    else:
        print("Some pieces could not be fitted.")

    #Convert image to color image
    puzzle = cv.cvtColor(puzzle, cv.COLOR_BGR2RGB)

    return notMatchedPieces, puzzle

    #####


###################################################################################################################
