"""
@author: Muamer Hrncic, Daniel Holzfeind, Alexandra Wissiak
license: MIT
"""
import cv2 as cv
#for usage, depending on which module you have installes either use 
from pylsd import lsd
#in pylsd-nova, pylsd, or use
#from pylsd.lsd import lsd
#for ocrd-fork-pylsd

import numpy as np
from numba import jit

def get_HOG_descriptor(winSize=100, blockSize=10, blockStride=5, cellSize=10, nbins=9, derivAperture=1,
                       winSigma=-1., histogramNormType=0, L2HysThreshold=0.2, gammaCorrection=1, nlevels=64,
                       signedGradient=True):
    """
    Method which returns the open cv implementation of the HOG descriptor.

    Attributes
    ----------
    winSize: int, optional
        Sets winSize with given value.
    blockSize: int, optional
        Sets blockSize with given value.
    blockStride: int, optional
        Sets blockStride with given value.
    cellSize: int, optional
        Sets cellSize with given value.
    nbins: int, optional
        Sets nbins with given value.
    derivAperture: int, optional
        Sets nbins with given value.        
    winSigma: double, optional
        Sets nbins with given value.        
    histogramNormType: int, optional
        Sets nbins with given value.        
    L2HysThreshold: double, optional
        Sets nbins with given value.    
    gammaCorrection: int, optional
        Sets nbins with given value.        
    nlevels: int, optional
        Sets nbins with given value.        
    signedGradient: boolean, optional
        Sets nbins with given value.
        
    Returns
    -------
        cv.HOGDescriptor(params)
    """


    return cv.HOGDescriptor((winSize, winSize), (blockSize, blockSize), (blockStride, blockStride), (cellSize, cellSize),
                            nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)


def hl_lines_and_connect(image, thres_comp, comp_boxes, minarea=45, kernelsize = 11):
    """
    Method which highlights the connection lines of components

    Attributes
    ----------
    image: numpy.array
        Input image.
    thres_comp: int
        Threshold image.
    comp_boxes: list
        List which contains the bounding boxes of the classified components.
    minarea: double, optional
        Minimum area for found contours.
    kernelsize: int, optional
        Size of the kernel used for closing.
        
    Returns
    -------
    out:
        Output image.
    line_endpoints:
        Endpoint coordinates of found connection lines. 
    """
    #Get copy of image
    out = image.copy()
    #Get rectangular kernel for image closing
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelsize, kernelsize))

    #Remove all found components from the threshold image
    for ((x, y, w, h), _) in comp_boxes:
        thres_comp[y: y + h, x: x + w] = 0

    #Apply morphological operation of closing to the threshold picture
    closed = cv.morphologyEx(thres_comp, cv.MORPH_CLOSE, kernel)
    #Get the contours
    contours = cv.findContours(
        closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    #List to hold the contours
    line_contour = []
    #List to hold the endpoints
    line_endpoints = []
    #Actual line
    nrActLine = 0

    #Main Loop
    for contour in contours:
        if cv.contourArea(contour) > minarea:
            #Append line segment
            line_contour.append((nrActLine, contour))
            #Create line mask
            mask = np.zeros(thres_comp.shape, np.uint8)
            #Draw line mask
            cv.drawContours(mask, [contour], 0, (255, 255, 255), 3)
            #Create random color to distinguish line segments
            color = (np.random.randint(0, 255), np.random.randint(
                0, 255), np.random.randint(0, 255))
            #Draw line into original picture
            cv.drawContours(out, [contour], 0, color, 3)
            #Thin lines
            thinned = ZhangSuen_thin(mask)
            #Get endpoints of lines
            endpoints = get_endpoints(thinned)
            #Store number of endpoints
            n = endpoints[0].size
            #Store line endpoints for every line i
            for j in range(n):
                #Get x and y coordinate of the j-th endpoint
                x, y = (endpoints[1][j], endpoints[0][j])
                #Store the actual line number and the endpoints
                line_endpoints.append([nrActLine, [x, y]])
            #increase actual line number
            nrActLine += 1

    return out, line_endpoints


def get_bounding_rect(img_reduced, border=10, minarea=75):
    """
    Method which highlights the connection lines of components

    Attributes
    ----------
    img_reduced: numpy.array
        Input image of the reduced circuit.
    border: int, optional
        Value for additional border.
    minarea: double, optional
        Minimal area for found contours.

    Returns
    -------
    bndg_rectangles:
        Bounding rectangles of found components used for classification.
    """
    #List to hold all bounding rectangles of eventuall componentes
    bndg_rectangles = []
    # Find remaining parts of components through contours in reduced image
    contours = cv.findContours(
        img_reduced, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    #Iterate over all contours
    for contour in contours:
        #Calculate area of contour
        area = cv.contourArea(contour)
        #If it is not large enough discard the contour
        if area < minarea:
            continue
        else:
            #Otherwise get the bounding rectangle
            x, y, w, h = cv.boundingRect(contour)
            #Get the longer side of the rectangle
            longer_side = max(w, h)
            #Recalculate the x and y coordinates of the bounding rectangle w.r.t the longer side
            x = int((2 * x + w - longer_side)/2)
            y = int((2 * y + h - longer_side)/2)
            #Append the bounding rectangle, by setting a new origin and new margin, defined by the
            #parameter border

            bndg_rectangles.append(
                [x - border, y - border, x + border + longer_side, y + border + longer_side])

    #Return all created rectangles
    return bndg_rectangles


def remove_lines(thres_line, comp_boxes, vertlow=80, verthigh=100, horlow=10, horhigh=170, kernelsize=11):
    """
    Method which remove the connection lines of components.

    Attributes
    ----------
    thres_line: numpy.array
        Input image of the reduced circuit.
    comp_boxes: list
        List which contains the bounding boxes of the classified components.
    vertlow: double, optional
        Minimum angle for vertical lines.
    verthigh: double, optional
        Maximum angle for vertical lines.
    horlow: double, optional
        Minimum angle for horizontal lines.
    horhigh: double, optional
        Maximum angle for horizontal lines.   
    kernelsize: int, optional
        Size of the kernel used for closing.     
        
    Returns
    -------
    Image with removed lines.
        
    """
    #Remove found line components from threshold image
    for ((x, y, w, h), _) in comp_boxes:
        thres_line[y:y+h, x:x+w] = 0
        
    cv.imwrite("stages/remove_lines.png", thres_line)

    #Detect all lines in the reduced threshold image using the line segment detection module from pylsd
    lines = lsd(thres_line)
    #Remove all lines which satisfy a certain angle condition.
    for line in lines:
        #Save endpoint coordinates of the line
        xs, ys, xe, ye, _ = line
        #Calculate delta x
        dx = xs - xe
        #Calculate delta x
        dy = ys - ye
        #Get the angle of the line in the correct quadrant.We look for angles in degrees.
        ang = np.abs(np.rad2deg(np.arctan2(dy, dx)))

        #If the angle of a line is not in the strip (horlow,horhigh) it is maybe a horizontal line
        #or if the angle of a line is in the strip (vertlow, verthigh) it is maybe a vertical line.
        if (ang < horlow or horhigh < ang or (vertlow < ang < verthigh)):
            #If it is a horizontal line or a vertical, remove the line by setting
            #all pixel on the line of width = 6 to black.
            cv.line(thres_line, (int(xs), int(ys)),
                    (int(xe), int(ye)), (0, 0, 0), 6)

    #Get kernel for the closing operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelsize, kernelsize))

    #Return the closed image
    return cv.morphologyEx(thres_line, cv.MORPH_CLOSE, kernel)


def draw_classified_comp(img, comp_boxes, border=5):
    """
    Method which draws the bounding boxes of classified components.

    Attributes
    ----------
    img: numpy.array
        Input image of the circuit.
    comp_boxes: list
        List which contains the bounding boxes of the classified components.
    border: int, optional
        Size of additional border.        
    """
    #Initialize component classes
    comp_class = ['Source', 'Capacitor',
                  'GND', 'Diode', 'Resistor', 'Inductor']
    #Set text font
    ft = cv.FONT_ITALIC
    
    for ((x, y, w, h), comp) in comp_boxes:
        #Write the class text to the component
        cv.putText(img, comp_class[comp], (x - border, y +
                    5 * border + h), ft, 0.72, (250, 0, 0), 1, cv.LINE_AA)
        if(comp in range(3)):
            #For every found components draw a bounding rectangle
            cv.rectangle(img, (x, y), (x+w, y+h), (243, 255, 60), 2)
        else:
            #For every found components draw a bounding rectangle
            cv.rectangle(img, (x, y), (x+w, y+h), (50, 60, 240), 2)


def sort_points(pts):
    """
    Method which sorts the corner points of the bounding box of line components.

    Attributes
    ----------
    pts: numpy.array
        Corner points of the bounding box/rectangle.
        
    Returns
    -------
    Array of sorted corner points.
             
    """
    #Store number of points
    n = len(pts)
    #Sort the point array with respect to the x-coordinate
    sort_x = pts[np.argsort(pts[:, 0]), :]
    #Store the first and the last point of the sorted list
    left_point = sort_x[:2, :]
    right_point = sort_x[n-2:, :]
    #Sort again by y coordinate
    (left_top, left_bottom) = left_point[np.argsort(left_point[:, 1]), :]
    (right_top, right_bottom) = right_point[np.argsort(right_point[:, 1]), :]
    #return sorted points
    return np.array([left_top, right_top, right_bottom, left_bottom])


def classify_gnd(actindex, found, lines, bordersmall=10, borderbig=15):
    """
    Method which classifies a line object as ground component.

    Attributes
    ----------
    actindex: numpy.array
        Index of the actual line
    found: numpy.array
        Array of indices of compatible lines.
    lines: numpy.array
        Array which hold the coordinates of start and endpoints of compatible lines.
    bordersmall: int, optional
        Additinal border for bounding box.
    borderbig: int, optional
        Additinal border for bounding box.
        
    Returns
    -------
    List containing bounding box of component and component class.
             
    """
    #Save indices of interest
    i, j, k = actindex, found[0][0], found[0][1]

    #Store the start and end points of the i-th and j-th line
    xs_i = lines[i, 0]
    ys_i = lines[i, 1]
    xe_i = lines[i, 2]
    ye_i = lines[i, 3]
    xs_j = lines[j, 0]
    ys_j = lines[j, 1]
    xe_j = lines[j, 2]
    ye_j = lines[j, 3]
    xs_k = lines[k, 0]
    ys_k = lines[k, 1]
    xe_k = lines[k, 2]
    ye_k = lines[k, 3]

    #Store coordinates into an array
    pts_to_sort = np.array([(xs_i, ys_i), (xe_i, ye_i),
                           (xs_j, ys_j), (xe_j, ye_j), (xs_k, ys_k), (xe_k, ye_k)])
    #Get corner point of bounding box, clockwise direction
    left_top, right_top, right_bottom, left_bottom = sort_points(pts_to_sort)

    #Add border to found box
    left_top = [left_top[0] - bordersmall, left_top[1] - bordersmall]
    right_top = [right_top[0] + bordersmall, right_top[1] - bordersmall]
    right_bottom = [right_bottom[0] + borderbig, right_bottom[1] + borderbig]
    left_bottom = [left_bottom[0] - borderbig, left_bottom[1] + borderbig]

    #return the bounding box coordinates and the componente class
    return [cv.boundingRect(np.array([left_top, right_top, right_bottom, left_bottom])), 2]


def classify_src_cap(actindex, found, lines, min_ratio=1, max_ratio=1.2, border=5):
    """
    Method which classifies a line object as source or capacitor component.

    Attributes
    ----------
    actindex: numpy.array
        Index of the actual line
    found: numpy.array
        Array of indices of compatible lines.
    lines: numpy.array
        Array which hold the coordinates of start and endpoints of compatible lines.
    min_ratio: double, optional
        Minimum value for length ratio of the two compatible lines.
    max_ratio: double, optional
        Maximum value for length ratio of the two compatible lines.
    border: int, optional
        Additinal border for bounding box.
        
    Returns
    -------
    List containing bounding box of component and component class.
             
    """
    
    #Save indices of interest
    i, j = actindex, found[0][0]

    #Store the start and end points of the i-th and j-th line
    xs_i = lines[i, 0]
    ys_i = lines[i, 1]
    xe_i = lines[i, 2]
    ye_i = lines[i, 3]
    xs_j = lines[j, 0]
    ys_j = lines[j, 1]
    xe_j = lines[j, 2]
    ye_j = lines[j, 3]

    #Store coordinates into an array
    pts_to_sort = np.array(
        [(xs_i, ys_i), (xe_i, ye_i), (xs_j, ys_j), (xe_j, ye_j)])
    #Get corner point of bounding box, clockwise direction
    left_top, right_top, right_bottom, left_bottom = sort_points(pts_to_sort)

    #Calculate lengths of the lines
    length_i = np.sqrt((xs_i - xe_i) ** 2 + (ys_i - ye_i) ** 2)
    length_j = np.sqrt((xs_j - xe_j) ** 2 + (ys_j - ye_j) ** 2)

    #Sort the lines by length
    line_short, line_long = sorted([length_i, length_j])
    #Calculate the ratio of the line lengths
    ratio = line_long/line_short

    #If the ratio is approximately 1, then the component is a capacitor
    if (min_ratio <= ratio <= max_ratio):
        comp = 1
    #otherwise it is a source
    elif (max_ratio < ratio):
        comp = 0

    #Add border to found box
    left_top = [left_top[0] - border, left_top[1]-border]
    right_top = [right_top[0] + border, right_top[1] - border]
    right_bottom = [right_bottom[0] + border, right_bottom[1] + border]
    left_bottom = [left_bottom[0] - border, left_bottom[1] + border]

    #return the bounding box coordinates and the componente class
    return [cv.boundingRect(np.array([left_top, right_top, right_bottom, left_bottom])), comp]


def get_pair_table(n, lines, tol=60):
    """
    Method which determines if some lines are compatible and how many compatible there are for one line.

    Attributes
    ----------
    n: int
        Number of lines.
    lines: numpy.array
        Array which hold the coordinates of start and endpoints of compatible lines.
    tol: double, optional
        Tolerance for the line lenght
        
    Returns
    -------
    pair_table:
        Table indicating compatible lines.
             
    """
    #Initialize emtpy table of pairs with -1, where pair_table[i,j] = -1 means, j is not a compatible line
    #for line i
    pair_table = np.zeros((n, n), dtype=np.int) - 1

    #Main loop
    for i in range(n):
        #Store middle point of the i-th line
        xmean_i, ymean_i = (
            (lines[i, 0] + lines[i, 2]) / 2, (lines[i, 1] + lines[i, 3]) / 2)
        #Iterate over all other lines
        for j in range(i + 1, n):
            #Store middle point of the i-th line
            xmean_j, ymean_j = (
                (lines[j, 0] + lines[j, 2]) / 2, (lines[j, 1] + lines[j, 3]) / 2)
            #Compute length of the vectorial difference of line i and line j
            length = np.sqrt((xmean_i - xmean_j) ** 2 +
                             (ymean_i - ymean_j) ** 2)
            #Check wether the length is smaller than the tolerance
            if length < tol:
                #Get the indices of entry equal to index of line j
                cand_j = np.where(pair_table == j)
                #If there are no j-candidates
                if len(cand_j[0]) == 0:
                    #Search for i-candidates
                    cand_i = np.where(pair_table == i)
                    #If there are no i-candidates
                    if len(cand_i[0]) == 0:
                        #Store the index j to the (i,j)-th entry
                        pair_table[i, j] = j
                    #otherwise if there are entries
                    elif len(cand_i[0]) > 0:
                        #Store the index j
                        pair_table[cand_i[0][0], j] = j
                else:
                    #Otherwise store the index i
                    pair_table[cand_j[0][0], i] = i

    return pair_table


def get_line_comp_boxes(lines):
    """
    Method which classify line components.

    Attributes
    ----------
    lines: numpy.array
        Array which hold the coordinates of start and endpoints of compatible lines.
        
    Returns
    -------
    components:
        List containing bounding box of component and component class.
             
    """
    #Get number of lines
    n = len(lines)
    #List to store classified line componets
    components = []

    #Get table of compatible lines
    pair_table = get_pair_table(n, lines)

    for i in range(n):
        #Indices of compatible lines
        found = np.where(pair_table[i, :] != -1)
        #Two found lines can only be classified as source or capacitor
        if len(found[0]) == 1:
            #Classify the object and append it to the list components
            components.append(classify_src_cap(i, found, lines))
        #Three found lines can be classified as ground component
        if len(found[0]) == 2:
            components.append(classify_gnd(i, found, lines))

    #Return found components
    return components


def get_ptl_distance(poss_point, line):
    """
    Method to calculate the point to line distance of the endpoint of a possible line point to a line.

    Attributes
    ----------
    poss_point: numpy.array
        Endpoint of a possible line components
        
    line: numpy.array
        Array which hold the coordinates of startpoint and endpoint of a line.
        
    Returns
    -------
    point to line distance:
        Either the normal distance or the minimal distance of the point to the line.
             
    """
    #Save point coordinates
    px = poss_point[0]
    py = poss_point[1]
    #Difference of the start point and the possible point
    sp_diff = line[0] - poss_point
    #Difference of the end point and the possible point
    ep_diff = line[1] - poss_point

    #Get the minimal distance of the possible point to the line
    ep_dist = min(np.linalg.norm(sp_diff), np.linalg.norm(ep_diff))
    #Transform line to vector which describes the direction of the line
    direction = line[1] - line[0]
    #Save line endpoints
    (xs, ys, xe, ye) = np.reshape(line, 4)

    #Get the coordinates of the normalized the direction
    (x_ndir, y_ndir) = direction / np.linalg.norm(direction)

    #Get the normalized cross product of the start point of the line and the possible point
    ncross = np.linalg.norm(np.cross(direction, sp_diff))
    #Calculate the normal distance to the direction
    normDist = ncross / np.linalg.norm(direction)

    #Calculate the dot product of the difference of the line and the possible point
    diff = (x_ndir * (px - xe)) + (y_ndir * (py - ye))

    #Calculate the coordinates of the intersecting line segment
    x_inter = (x_ndir * diff) + xe
    y_inter = (y_ndir * diff) + ye

    #Variable for x condition
    on_x = False
    #Variable for y condition
    on_y = False

    #Check if the intersection point is on the original line segment
    if((xe <= x_inter <= xs) or (xs <= x_inter <= xe)):
        on_x = True
    if((ys <= y_inter <= ye) or (ye <= y_inter <= ys)):
        on_y = True
    #Check if both intersection conditions are true
    if on_x and on_y:
        #Return the normalized distance
        return normDist
    else:
        #otherwise return distance to the line endpoints
        return ep_dist


def get_lines_single(linelist, act_line, img_ep, length, y_min, y_max, x_min, x_max, border=5, tol=0.7):
    """
    Method to calculate the point to line distance of the endpoint of a possible line point to a line.

    Attributes
    ----------
    linelist: numpy.array
        List of lines. 
    act_line: numpy.array
        Array which hold the coordinates of startpoint and endpoint of a line.
    img_ep: numpy.array
        Endpoint image.
    length: double
        Length of the line.
    y_min: int
        Minimum y coordinate.
    y_max: int
        Maximum y coordinate.
    x_min: int
        Minimum x coordinate.
    x_max: int
        Maximum x coordinate.
    border: int, optional
        Additional border
    tol: double, optional
        Tolerance for appending new lines         
    """
    #Get endpoints of lines close to the actual line
    ep_id = np.where(img_ep[y_min - border: y_max +
                     border, x_min - border: x_max + border] == 255)

    #Inititialize counter for valid points
    count = 0

    #Store number of endpoint indices
    m = len(ep_id[0])

    #Iterate over all pairs
    for k in range(m):
        #Take the next pair of endpoints
        poss_point = np.array(
            [x_min - border + ep_id[1][k], y_min - border + ep_id[0][k]])
        #Calculate the distance of the point to the actual line
        ptl_dist = get_ptl_distance(poss_point, act_line)
        #If the distance of the point to the line is smaller than two, then this point
        #is a possible candidate
        if(ptl_dist < 2):
            count += 1

    #If the number of points on the considered line is greater then the reduced actual line, append the line
    if count > length * tol:
        linelist.append(np.reshape(act_line, 4))


def get_lines(ends, vertlow=80, verthigh=100, horlow=10, horhigh=170, minlen=5, maxlen=100, border=5):
    """
    Method to get vertical and horizontal lines of possible line components.

    Attributes
    ----------
    ends: numpy.array
        Endpoints in the image with marked endpoints.
        
    vertlow: double, optional
        Minimum angle for vertical lines.
    verthigh: double, optional
        Maximum angle for vertical lines.
    horlow: double, optional
        Minimum angle for horizontal lines.
    horhigh: double, optional
        Maximum angle for horizontal lines.
    minlen: double, optional
        Minimum length for lines.
    maxlen: double, optional
        Maximum length for lines.
    border: int, optional
        Additional border.

    Returns
    -------
    (verticals, horizontal):
        Coordinates of vertical and horizontal lines of possible line elements.
             
    """
    #List to hold coordinates of vertical lines from point (x1,y1) to (x2,y2)
    verticals = []
    #List to hold coordinates of vertical lines from point (x1,y1) to (x2,y2)
    horizontals = []
    #Get image where enpoint pixels have value 11
    img_ep = cv.imread("stages/endpoints.png", 0)
    #Store number of lines
    nrlines = ends[0].size

    #Main loops
    for i in range(nrlines):
        #Get coordinates of start point
        xs, ys = (ends[1][i], ends[0][i])
        #Iterate over every other line
        for j in range(i + 1, nrlines):
            #Get coordinates of end point
            xe, ye = (ends[1][j], ends[0][j])

            #Save the actual line
            act_line = np.array([[xs, ys], [xe, ye]])

            #Get line length
            length = np.sqrt((xs - xe)**2 + (ys - ye)**2)
            #Get the min and max values of the coordinates
            y_min, y_max = sorted([ys, ye])
            x_min, x_max = sorted([xs, xe])

            #Get the angle of the line in the correct quadrant.We look for angles in degrees.
            ang = np.abs(np.rad2deg(np.arctan2(ys - ye, xs - xe)))

            #Consider either vertical lines fulfilling the vertical angle condition
            if ((vertlow < ang < verthigh) and (minlen < length < maxlen)):

                #Get possible candidates for line components
                get_lines_single(verticals, act_line, img_ep,
                                 length, y_min, y_max, x_min, x_max)

            #or consider horizontal lines, fulfilling the hoizontal angle condition
            elif ((ang < horlow or horhigh < ang) and (minlen < length < maxlen)):
                #Get possible candidates for line components
                get_lines_single(horizontals, act_line, img_ep,
                                 length, y_min, y_max, x_min, x_max)

    #Return endpoints of vertical and horizontal lines
    return np.array(verticals), np.array(horizontals)


def get_line_comp(endpoints):
    """
    Main method to get vertical and horizontal lines of possible line components.

    Attributes
    ----------
    endpoints: numpy.array
        Endpoints in the image with marked endpoints.

    Returns
    -------
    comp_boxes:
        List containing bounding box of component and component class.
    """
     
    #Get the endpoints of the vertical lines and the horizontal lines
    vertical_lines, horizontal_lines = get_lines(endpoints)
    #Try to detect vertical and horizontal components and store the bounding boxes
    #to the global placeholder for all components
    comp_boxes = get_line_comp_boxes(
        vertical_lines) + get_line_comp_boxes(horizontal_lines)
    return comp_boxes


def get_endpoints(threshold, kernelsize=3):
    """
    Method to mark endpoints of possible line components.

    Attributes
    ----------
    threshold: numpy.array
        Threshold image of circuit.
    kernelsize: int, optional
        Size of the kernel for filtering operation.

    Returns
    -------
    endpoints:
        List containing bounding box of component and component class.
    """
    #Make copy of input
    skel = threshold.copy()
    #Set non black pixels to one
    skel[skel != 0] = 1
    #Convert to uint8
    skel = np.uint8(skel)

    #Get 3x3 rectangular kernel for filter process
    kernel = np.uint8(cv.getStructuringElement(
        cv.MORPH_RECT, (kernelsize, kernelsize)))
    #Give the center point a big weight, which means that 
    kernel[1, 1] = 21
    #Filter image with given kernel. Applying the filter gives
    #an image where possible line endpoints or component endpoints have values
    #equal to 21 and cont. line points or branch points have values > 21.
    #So filtering mark endpoints.
    img_filter = cv.filter2D(skel, -1, kernel)

    #Get indices of the endpoints, namely those pixels with pixelvalue 21
    endpoints = np.where(img_filter == 22)
    return endpoints

#Use Just In Time compiler, to improve performance of single iteration


@jit
def ZhangSuen_thin_single(image, step):
    """
    Single iteration of thinning the lines of a threshold image using the method
    proposed by Zhang and Suen.

    Attributes
    ----------
    image: numpy.array
        Threshold image of circuit.
    step: int
        Actual thinning step.

    Returns
    -------
    thinned image:
        Bitwise AND of the original image and the determined mask.
    """
    #Single iteration of Zhang Suen thinning algorithm, without checking if center is black
    mask = np.zeros(image.shape, np.uint8)
    h, w = image.shape

    #Main Loops
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            #Get pixel neighbours
            P2 = image[i - 1, j]
            P3 = image[i - 1, j + 1]
            P4 = image[i, j + 1]
            P5 = image[i + 1, j + 1]
            P6 = image[i + 1, j]
            P7 = image[i + 1, j - 1]
            P8 = image[i, j - 1]
            P9 = image[i - 1, j - 1]

            #Number of transitions from white to black
            A = (P2 == 0 and P3 == 1) + (P3 == 0 and P4 == 1) + \
                (P4 == 0 and P5 == 1) + (P5 == 0 and P6 == 1) + \
                (P6 == 0 and P7 == 1) + (P7 == 0 and P8 == 1) + \
                (P8 == 0 and P9 == 1) + (P9 == 0 and P2 == 1)

            #Number of black pixel neighbours
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

            #Check which step is active
            if(step == 1):
                #At least one of P2 and P4 and P6 is white
                nr_white_1 = (P2 * P4 * P6)
                #At least one of P4 and P6 and P8 is white
                nr_white_2 = (P4 * P6 * P8)
            else:
                nr_white_1 = (P2 * P4 * P8)
                #At least one of P2 and P6 and P8 is white
                nr_white_2 = (P2 * P6 * P8)
            #Check all conditions.
            if A == 1 and 2 <= B <= 6 and nr_white_1 == 0 and nr_white_2 == 0:
                mask[i, j] = 1

    #Return bitwise and of image and mask
    return image.astype(np.uint8) & ~mask


def ZhangSuen_thin(img):
    """
    Main method of thinning the lines of a threshold image using the method
    proposed by Zhang and Suen.

    Attributes
    ----------
    image: numpy.array
        Threshold image of circuit.

    Returns
    -------
    act_img:
        Thinned image.
    """
    #Zhang-Suen thinning algorithm
    #https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm

    #Get copy of image
    act_img = img.copy()
    #Set white pixels to one
    act_img[act_img == 255] = 1
    #Initialize prev_image with zero entries
    prev_img = np.zeros(img.shape[:2], np.uint8)
    #Initialize variable for image difference of each iteration
    diff_img = None
    #Variable to indicate if thinning is finished
    thinned = False

    #Main loop
    while not thinned:
        #Step 1
        act_img = ZhangSuen_thin_single(act_img, 1)
        #Step 2
        act_img = ZhangSuen_thin_single(act_img, 2)
        #Compute the pixelwise difference of iteration i and i-1
        diff_img = np.absolute(act_img - prev_img)
        #Store previous image
        prev_img = act_img.copy()
        #If the actual difference only contains zeros stop the iteration
        if not diff_img.any():
            thinned = True

    #Rescale 1-pixels to white pixels and return thinned image
    act_img[act_img == 1] = 255
    #Return image
    return act_img


def image_preproc(src, blurkernelsize=9, blocksize=7, const=2, morphIterations=2, kernelsize=3):
    """
    Method to preprocess images of circuit sketches.
    
    Attributes
    ----------
    src: numpy.array
        Source image.
        
    blurkernelsize: int, optional
        Size of the kernel used for Gaussian blurring
        
    blocksize: int, optional
        Size of blocks for adaptive thresholding.
        
    const: int, optional
        Constant for adaptive thresholding. 
        
    morphIterations: int, optional
        Number of dilating iterations
        
    kernelsize: int, optional
        Size of the kernel used for dilating.

    Returns
    -------
    thinned, thres_line, thres_comp:
        Images used for further steps.
    endpoints:
        Endpoints of possible line components.
    """
    #Convert source image to grayscale
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #Apply Gaussian filter to denoise image with a 9x9 kernel
    img = cv.GaussianBlur(gray, (blurkernelsize, blurkernelsize), 0)
    #Apply adaptive thresholding
    thres_line = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blocksize, const)
    #Make copy of threshold image
    thres_comp = thres_line.copy()
    #Get kernel for morphological  operations
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernelsize, kernelsize))
    #Dilate image
    dilated = cv.dilate(thres_line, kernel, iterations=morphIterations)
    #Thin image with Zhang-Suen thinning algorithm
    thinned = ZhangSuen_thin(dilated)
    #Save skeleton of image
    cv.imwrite("stages/endpoints.png", thinned)
    #Save threshold image
    cv.imwrite("stages/threshold.png", thres_comp)
    #Get skeleton points
    endpoints = get_endpoints(thinned)
    return thinned, thres_line, thres_comp, endpoints
