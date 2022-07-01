# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""
import cv2 as cv

###################################################################################################################


def getKernel(kernelType='rect', ksize_x=3, ksize_y=3):
    """
    Method to get different kernels.
        
    Attributes
    ----------
    kernelType: string
        Possible values for kernelType in {'rect','ellipse','cross'}.
    ksize_x: int
        Kernel size in x direction.
    ksize_y: int
        Kernel size in y direction.
        
    Returns
    -------
    kernel: cv.getStructuringElement(kernelType, size=(ksize1,ksize2))

    """
    if(not(kernelType in ['rect', 'ellipse', 'cross'])):
        print('Given kernelTyp is not in the list!')
        print('A rect. kernel of size 3x3 will be returned!')
        return cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    #rectangular kernel
    if(kernelType in 'rect'):
        return cv.getStructuringElement(cv.MORPH_RECT, (ksize_x, ksize_y))
    #eliptical kernel
    if(kernelType in 'ellipse'):
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksize_x, ksize_y))
    #cross-like kernel
    if(kernelType in 'cross'):
        return cv.getStructuringElement(cv.MORPH_CROSS, (ksize_x, ksize_y))


def enhanceSharpnes(img, kernel, nrIter=10, dilIter=1, erodIter=1):
    """
    Iterative grayscale morphological images processing algorithm
    to enhance sharpness of structures in a blurry image.
        
    Attributes
    ----------
    img: Numpy array, type uint8
        Image to be processed.
    kernel: cv.getStructuringElement(kernelType, size=(ksize1,ksize2))
        Kernel used for processing.
    nrIter: int
        Number of processing iterations.
    dilIter: int
        Number of dilations.
    erodIter: int
        Number of erosions.
        
    Returns
    -------
    img_enh: Numpy array, type uint8
        Enhanced image.
    """
    #image to be returned
    img_enh = img.copy()

    #save image shape
    rows, cols = img.shape

    #main algorithm
    for i in range(nrIter):
        #dilate the image once with the given kernel and save it
        Im_d = cv.dilate(img_enh, kernel, iterations=dilIter)

        #erode the image once with the given kernel and save it
        Im_e = cv.erode(img_enh, kernel, iterations=erodIter)

        #save the "mean" of the dilated and the eroded image
        Im_h = 0.5 * (Im_d + Im_e)

        for j in range(rows):
            for k in range(cols):
                #if the pixel value of the original is greater than the mean
                if(img_enh[j, k] > Im_h[j, k]):
                    #save pixel value of the dilated image
                    img_enh[j, k] = Im_d[j, k]
                else:
                    #save the pixel value of the eroded image
                    img_enh[j, k] = Im_e[j, k]

    return img_enh

###################################################################################################################
