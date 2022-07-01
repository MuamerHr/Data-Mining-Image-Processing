# -*- coding: utf-8 -*-
"""
@author: Freislich, Hrncic, Siebenhofer
License: MIT
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def checkValue(r,radius):
    """

    Parameters
    ----------
    r : avg intensity of the red channel of the detected coin
    radius : radius of the coin

    Returns
    -------
    TYPE monetary values of the coin

    """
    if(r>160):
        # if high red intensity -> 1-5 cent depending on radius
        if(radius <42):
            return 0.01
        if(radius >42 and radius <50):
            return 0.02
        if(radius >=50):
            return 0.05
    elif(r<=160 and r>125):
        # medium red intensity -> 10-50 cent depending on radius
        if(radius <55):
            return 0.10
        if(radius >=55 and radius <60):
            return 0.20
        if(radius >=60):
            return 0.50
    elif(r<=125 and r>110):
        # red intensity in a specific range -> 2 Euro
        return 2.00
    elif(r<=110 and r>100):
        # red intensity in a specific range -> 1 Euro
        return 1.00
    else:
        return 0.00
             
def checkCoin(x_offset,y_offset,radius, number):
    """
    

    Parameters
    ----------
    x_offset : x Position of the center of the coin
    y_offset : y Positon of the center of the coin
    radius : radius of the coin
    number : number that shall be assigned to the coin

    Returns
    -------
    value : monetary value of the coin

    """
    r=0
    g=0
    b=0
    amount = 0
    # get the avg color of the detected circle
    for x in range(x_offset-radius,x_offset+radius):
        for y in range(y_offset-radius,y_offset+radius):
            if((y-y_offset)*(y-y_offset)+(x-x_offset)*(x-x_offset)<=(radius*radius)):
                amount+=1
                r+=input_img[y,x,0]
                g+=input_img[y,x,1]
                b+=input_img[y,x,2]

    text = "["+str(int(r/amount))+","+str(int(g/amount))+ ","+str(int(b/amount))+"]"            
    print("Circle Nr:" + str(number) + "\n" + text)
    # offset for disply of text
    position = (x_offset-20,y_offset+4)
    """
    By analyzing the values of the different color channels of the different 
    coins we found out, that is was the easiest way to differentiate the coins
    by looking at the intensity of the red channel. 
    """
    value = checkValue(int(r/amount),radius)
    # console display of amount for testing
    print(checkValue(int(r/amount),radius))
    # write the values of the coin ontop of it 
    cv2.putText(oimg, str(value),position,cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0), 2)
    return value            
# img loading and graying for algorithms
oimg  = cv2.cvtColor(cv2.imread("coins.jpg"),cv2.COLOR_BGR2RGB)
# copy for output
input_img = oimg.copy()
img = cv2.cvtColor(oimg,cv2.COLOR_RGB2GRAY)
# medianblur for noise reduction 
img = cv2.medianBlur(img,7)
# HoughCircles with carefully chosen parameters to detect all the coins
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,120,
                            param1=50,param2=30,minRadius=40,maxRadius=70)
circles = np.uint16(np.around(circles))
# variables for numbering and counting the amount
counter=0
sum_coins = 0
# make original image all white to display only the detected coins + values
oimg[:,:,:]=255
for i in circles[0,:]:
    counter+=1
    # draw the outer circle
    cv2.circle(oimg,(i[0],i[1]),i[2],(255,0,0),2)
    # check the detected coins    
    sum_coins+=checkCoin(i[0], i[1], i[2], counter)

# print sum of coins
print(str(sum_coins) + " Euro insgesamt")
output = "Number of coins: " + str(counter) + " , Sum: " + str(sum_coins) + " Euro "
cv2.putText(oimg,output ,(15,input_img.shape[0]-5 ),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 0, 0), 2)
# Show result
fig=plt.figure("Circle-Deteciton", figsize=(14,8))
plt.subplot(1,2,2),plt.imshow(oimg), plt.title("detected coins")
plt.subplot(1,2,1),plt.imshow(input_img), plt.title("original")
# write figure into img
fig.savefig("out/counting_coins.jpg")
