# to detect the shapes
import math
from math import isclose
import cv2
import numpy as np

def f(x):
    return (x)

def angle(pt0, pt1, pt2):
    # For A(x,y), B(x,y), C(x,y)
    #  vectorAB = B-A = ((Bx-Ax)-(By-Ay))
    #  vectorBC = C-B = ((Cx-Bx)-(Cy-By))
    # dx1 = pt1x-coordinate - pt0x-coordinate
    dx1 = pt0[0] - pt1[0]
    dy1 = pt0[1] - pt1[1]
    # vectorAB
    d1 = np.array([dx1, dy1])

    # vectorBC
    dx2 = pt1[0] - pt2[0]
    dy2 = pt1[1]  - pt2[1]
    d2 = np.array([dx2, dy2])

    # dot product of d1 en d2
    # dotPrd1d2= d2(x)*d1(x)+d2(y)*d1(y)
    dotPr = np.dot(d1, d2)
    # ||AB|| = sqrt(math.pow(vectorABx, 2) + math.pow(VectorABy,2))
    # ||BC|| = sqrt(math.pow(vectorBCx, 2) + math.pow(vectorBCy,2))
    lengthOfd1 = math.sqrt(math.pow(dx1,2) + math.pow(dy1, 2))
    lengthOfd2 = math.sqrt(math.pow(dx2, 2) + math.pow(dy2, 2))
    # returns the angle of the vector by arc cos alpha
    return np.arccos(dotPr/(lengthOfd1*lengthOfd2)) *180 / math.pi

def length(pt0, pt1, pt2, pt3):
    # vectorAB
    dx1 = pt0[0] - pt1[0]
    dy1 = pt0[1] - pt1[1]


    # vectorBC
    dx2 = pt1[0] - pt2[0]
    dy2 = pt1[1] - pt2[1]

    # vectorCD
    dx3 = pt2[0] - pt3[0]
    dy3 = pt2[1] - pt3[1]

    # vectorDA
    dx4 = pt3[0] - pt0[0]
    dy4 = pt3[1] - pt0[1]

    # ||AB|| = sqrt(math.pow(vectorABx, 2) + math.pow(VectorABy,2))
    # ||BC|| = sqrt(math.pow(vectorBCx, 2) + math.pow(vectorBCy,2))
    lengthOfd1 = math.sqrt(math.pow(dx1, 2) + math.pow(dy1, 2))
    lengthOfd2 = math.sqrt(math.pow(dx2, 2) + math.pow(dy2, 2))
    # ||CD|| = sqrt(math.pow(vectorCDx, 2) + math.pow(VectorCDy,2))
    # ||DA|| = sqrt(math.pow(vectorDAx, 2) + math.pow(vectorDAy,2))
    lengthOfd3 = math.sqrt(math.pow(dx3, 2) + math.pow(dy3, 2))
    lengthOfd4 = math.sqrt(math.pow(dx4, 2) + math.pow(dy4, 2))

    if (isclose(lengthOfd1, lengthOfd2, rel_tol=0.02) & isclose(lengthOfd2, lengthOfd3, rel_tol=0.02) &
            isclose(lengthOfd3, lengthOfd4, rel_tol=0.02) & isclose(lengthOfd4, lengthOfd1, rel_tol=0.02)):
        return True
    else:
        return False

# setting threshold of gray image
cv2.namedWindow("Threshold Control",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Threshold Control", 300, 300)
cv2.createTrackbar("Lower Thres", "Threshold Control", 0, 255, f)
cv2.createTrackbar("Upper Thres", "Threshold Control", 0, 255, f)
cv2.namedWindow("Area Control")
cv2.createTrackbar("Lower Area Size", "Area Control", 0, 10000, f)
cv2.createTrackbar("Upper Area Size", "Area Control", 0, 100000, f)
while(True):
    # reading image
    img = cv2.imread('warpcolored.jpg')
    # img = cv2.resize(img, (500,500))

    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold_lower = cv2.getTrackbarPos("Lower Thres", "Threshold Control")
    threshold_upper = cv2.getTrackbarPos("Upper Thres", "Threshold Control")
    _, threshold = cv2.threshold(gray, threshold_lower, threshold_upper, cv2.THRESH_BINARY)
    cv2.imshow("Threshold", threshold)
    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    # create a two dimensional list for id and value of each shape
    area_list = []
    # list for storing names of shapes
    try:
        for contour in contours:
            # ignoring first counter because
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue
            x1, y1 = contour[0][0]

            # to get the area of each contour
            area = cv2.contourArea(contour)

            lower_area_threshold = cv2.getTrackbarPos("Lower Area Size", "Area Control")
            upper_area_threshold = cv2.getTrackbarPos("Upper Area Size", "Area Control")
            if (lower_area_threshold < area < upper_area_threshold):
                new_area = [i, area]
                area_list.append(new_area)

                # cv2.approxPloyDP() function to approximate the shape
                approx = cv2.approxPolyDP(
                    contour, 0.01 * cv2.arcLength(contour, True), True)

                # using drawContours() function
                cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

                # finding center point of shape
                M = cv2.moments(contour)
                if M['m00'] != 0.0:
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
                    # print("i is " + str(i))

                # putting shape name at center of each shape
                if len(approx) == 3:
                    cv2.putText(img, 'Triangle' + str(i), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                elif len(approx) == 4:

                    x, y, w, h = cv2.boundingRect(contour)

                    # approx = approx[:, 0]

                    # to determine the coordinate
                    # for pts in approx:
                    #     cv2.putText(img, "{0},{1}".format(pts[0], pts[1]), (pts[0], pts[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                    ratio = float(w) / h
                    if length(approx[0][0], approx[1][0], approx[2][0], approx[3][0]) == True:
                        if abs(angle(approx[0][0], approx[3][0], approx[2][0]) - 90) <= 1:
                            cv2.putText(img, 'Square ID:' + str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                                        2)
                        else:
                            cv2.putText(img, 'Rhombus ID:' + str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                                        2)
                    else:
                        if abs(angle(approx[0][0], approx[1][0], approx[2][0]) - angle(approx[0][0], approx[3][0],
                                                                                       approx[2][0])) <= 1:
                            if abs(angle(approx[0][0], approx[3][0], approx[2][0]) - 90) <= 1:
                                cv2.putText(img, 'Rectangle ID:' + str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 255, 0),
                                            2)
                            else:
                                cv2.putText(img, 'Parallelogram ID:' + str(i), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (255, 255, 0), 2)

                        else:
                            cv2.putText(img, 'Quadrilateral ID:' + str(i), (x1-100, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 0),
                                        2)


                elif len(approx) == 5:
                    cv2.putText(img, 'Pentagon ID:' + str(i), (x-50, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                elif len(approx) == 6:
                    cv2.putText(img, 'Hexagon ID:' + str(i), (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                else:
                    cv2.putText(img, 'Circle ID:' + str(i), (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                i += 1
                min = area_list[0][1]
                max = area_list[0][1]
                min_id = 1
                max_id = 1
                for a in range(len(area_list)):
                    if area_list[a][1] < min:
                        min = area_list[a][1]
                        min_id = area_list[a][0]
                    if area_list[a][1] > max:
                        max = area_list[a][1]
                        max_id = area_list[a][0]
                print("the smallest element if is ", min_id, " with area of ", min)
                print("the largest element if is ", max_id, " with area of ", max)
                cv2.rectangle(img, (0, img.shape[1]+10), (440, img.shape[1]-25), (180, 135, 100), -1)
                cv2.putText(img, ("the smallest element id is "+ str(min_id)+ " with area of "+ str(min)),
                            (0, img.shape[1] - 12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(img, ("the largest element id is "+ str(max_id)+ " with area of "+ str(max)),
                            (0, img.shape[1]+3),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    except NameError:
        continue

    cv2.imshow('shapes', img)
    if cv2.waitKey(20) & 0xFF == ord('s'):  # Press 's' to save image
        print(cv2.imwrite("ShapenAreas.jpg", img))
    if cv2.waitKey(20) & 0xFF == 27: # press esc to exit
        break
cv2.destroyAllWindows()

