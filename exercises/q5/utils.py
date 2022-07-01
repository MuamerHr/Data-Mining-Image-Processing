# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""

import cv2 as cv
import numpy as np

def getEdges(image, rho=1, theta=1, threshold=100, minLineLength=10, maxGap=50):
    """
    Method to get edges through probabilistic Hough-transformation.

    Attributes
    ----------
    rho: double
        Distance resolution of the accumulator in pixels.
    theta: double
        Angle resolution of the accumulator in radians
    threshold: int
        Accumulator threshold parameter. Only those lines are returned that get enough votes
    minLineLength: int
        Minimum line length. Line segments shorter than that are rejected.
    maxGap: int
        Maximum allowed gap between points on the same line to link them.

    Returns
    -------
    lines: Array of int32
        Edges found by probabilistic Hough transformation

    """
    # find lines with hough transformation and return them
    return cv.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxGap)


def getRotation(x1, y1, x2, y2):
    """
    Method to get the rotation of a rectangular bounded picture.
    Rotation is obtained by getting rotation of a found line with
    coordinates of the defining points ((x1,y1),(x2,y2))
    with respect to the horizontal x axis.

    Attributes
    ----------
    x1: double
        x-coodinate of the first defining point of the line.
    y1: double
        y-coodinate of the first defining point of the line.
    x2: double
        x-coodinate of the second defining point of the line.
    y2: double
        y-coodinate of the second defining point of the line.

    Returns
    -------
    rho: double
        Rotation angle with respect to the horizontal x-axis.
    """
    xdif = np.abs(x1-x2)
    ydif = np.abs(y1-y2)

    # Determine if horizontal or vertical line (suitable for this example)
    if(xdif > ydif):
        isHorizontal = True
    else:
        isHorizontal = False

    # let(x1,y1) be the origin
    # we want to know if must rotate the image clockwise or anticlockwise
    if(isHorizontal):
        # assume rotation angle to be clockwise
        rho = np.arctan(ydif/xdif)
        # if y - coordinate of second endpoint greater than zero then
        # rotationangle anticlockwise
        if(y1 > y2):
            rho = -rho

    else:
        # assume rotation angle to be clockwise
        rho = np.arctan(xdif/ydif)
        # if x - coordinate of second endpoint smaller than zero then
        # rotationangle anticlockwise
        if(x2 > x1):
            rho = -rho

    return rho
