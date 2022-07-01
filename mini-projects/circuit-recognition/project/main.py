"""
@author: Muamer Hrncic, Daniel Holzfeind, Alexandra Wissiak
license: MIT
"""
import cv2 as cv
import imutils
import matplotlib.pyplot as plt

from electrical_comp_SVM import classify
from image_processing import image_preproc
from image_processing import get_line_comp
from image_processing import draw_classified_comp
from image_processing import remove_lines
from image_processing import get_bounding_rect
from image_processing import hl_lines_and_connect

def get_comp_templates():
    #Get the template images of supported electrical components.
    src = cv.imread("data/comp_templates/src.png")
    cap = cv.imread("data/comp_templates/c.png")
    gnd = cv.imread("data/comp_templates/gnd.png")
    diode = cv.imread("data/comp_templates/d.jpg")
    res = cv.imread("data/comp_templates/r.jpg")
    ind = cv.imread("data/comp_templates/i.jpg")
    return src, cap, gnd, diode, res, ind
    
if __name__ == "__main__":

    #############################################################################

    #Terminal Output
    print("Welcome to the electrical component classifier!")
    print("Type the path to the variable img_circuit.")
    print("Examples are in the folder examples.")
    print("The image can contain a combination of ")
    print("the following components as input parameter:")
    print("Source, Capacitor, Ground, Diode, Resistor, Inductor.")
    
    #Get template components    
    src, cap, gnd, diode, res, ind = get_comp_templates()

    #Show template components
    fig1 = plt.figure(num=1, figsize=(14,8))
    plt.subplot(231),plt.imshow(src),plt.title("Source")
    plt.subplot(232),plt.imshow(cap),plt.title("Capacitor")
    plt.subplot(233),plt.imshow(gnd),plt.title("Ground")
    plt.subplot(234),plt.imshow(diode),plt.title("Diode")
    plt.subplot(235),plt.imshow(res),plt.title("Resistor")
    plt.subplot(236),plt.imshow(ind),plt.title("Inductor")
    #Save image to output folder
    fig1.savefig("output/Templates.png")
    
    #############################################################################
    #Read image
    img_circuit = cv.imread("examples/circuit.png")

    # Resize image.
    img_circuit = imutils.resize(img_circuit, width=640)
    #Resize circuit_4
    #img_circuit = imutils.resize(img_circuit, width=500)
    # Make copy of the original image, for visualization
    org = img_circuit.copy()

    #############################################################################

    # First things first
    # Preprocess image and get image variants for the different steps
    #params for circuit, circuit_2, circuit_3
    thinned, thres_line, thres_comp, endpoints = image_preproc(img_circuit, blurkernelsize=7, blocksize=7, const=2, morphIterations=1, kernelsize=3)
    #params for circuit_1
    #thinned, thres_line, thres_comp, endpoints = image_preproc(img_circuit, blurkernelsize=11, blocksize=5, c=2, morphIterations=2, kernelsize=3)
    #params for circuit_4
    #thinned, thres_line, thres_comp, endpoints = image_preproc(img_circuit, blurkernelsize=3, blocksize=11, c=2, morphIterations=1, kernelsize=3)
    #params for single_comp
    #thinned, thres_line, thres_comp, endpoints = image_preproc(img_circuit, blurkernelsize=7, blocksize=11, c=2, morphIterations=1, kernelsize=5)
    
    
    #Show preprocessing steps
    fig2 = plt.figure(num=2, figsize=(14,8))
    plt.subplot(131),plt.imshow(img_circuit),plt.title("Original image")
    plt.subplot(132),plt.imshow(thres_line),plt.title("Threshold image")
    plt.subplot(133),plt.imshow(thinned),plt.title("Thinned image")
    fig2.savefig("output/0_Preprocessing.png")
    #############################################################################

    # First step:
    # 1) Try to detect line components source, ground, capacitor if possible.
    # 2) Store them to the global placeholder for all components
    comp_boxes = get_line_comp(endpoints)

    # Get a copy of the original picture
    img_circuit = org.copy()

    fig3 = plt.figure(num=3, figsize=(14,8))
    # Draw the rectangles of the classified components
    draw_classified_comp(img_circuit, comp_boxes, 0)
    # Plot the image of the classified components
    plt.subplot(111),plt.imshow(img_circuit),plt.title("Classified line components")
    fig3.savefig("output/1_line_comp.png")
    #############################################################################

    # Second step:
    # 1) Remove the line objects.
    # 2) Try to detect lines.
    # 3) Remove them from threshold image.
    # 4) Apply morphological closing to the image.
    # 5) Get an image which contains diodes, resistors and inductances,
    #   if present in the orig. image of the circuit.
    img_reduced = remove_lines(thres_line, comp_boxes)
    
    fig4 = plt.figure(num=4, figsize=(14,8))
    # Plot the reduce image
    plt.subplot(111),plt.imshow(img_reduced),plt.title("Remove lines, get remaining components")
    fig4.savefig("output/2_remove_lines.png")
    

    #############################################################################
    # Third Step:
    # 1) Try to find the contours of possible remaining components
    # 2) Get the bounding rectangles of the found contours w.r.t. to a minimum area
    bndg_rectangles = get_bounding_rect(img_reduced)

    #############################################################################

    # Fourth Step:
    # Try to classify the remaining components in the reduced threshold image
    # with the help of the found bounding rectangles, which play the role as region of interest
    comp_boxes = classify(thres_comp, bndg_rectangles, comp_boxes)

    # Get a copy of the original picture
    img_circuit = org.copy()

    fig5 = plt.figure(num=5, figsize=(14,8))
    # Draw the rectangles of all classified components
    draw_classified_comp(img_circuit, comp_boxes, 1)
    # Plot the image of the classified components
    plt.subplot(111),plt.imshow(img_circuit),plt.title("Remaining classified components")
    fig5.savefig("output/3_classified_comp.png")

    #############################################################################

    # Last step:
    fig6 = plt.figure(num=6, figsize=(14,8))
    # Highlight connected lines ang get connecting components
    img_out, connections = hl_lines_and_connect(
        img_circuit, thres_comp, comp_boxes)
    # Plot the image of the classified components
    plt.subplot(111),plt.imshow(img_out),plt.title("Classified components and highlighted lines")
    fig6.savefig("output/4_output.png")
