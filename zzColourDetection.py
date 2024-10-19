import cv2

import matplotlib.pyplot as plt


def f(x):
    return x


# creating trackbars allow users to adjust the boundaries
cv2.namedWindow("Threshold 1", cv2.WINDOW_NORMAL)
cv2.createTrackbar("R", "Threshold 1", 0, 255, f)
cv2.createTrackbar("G", "Threshold 1", 0, 255, f)
cv2.createTrackbar("B", "Threshold 1", 0, 255, f)
cv2.createTrackbar("Lower Area", "Threshold 1", 0, 1500, f)

cv2.namedWindow("Threshold 2", cv2.WINDOW_NORMAL)
cv2.createTrackbar("R", "Threshold 2", 0, 255, f)
cv2.createTrackbar("G", "Threshold 2", 0, 255, f)
cv2.createTrackbar("B", "Threshold 2", 0, 255, f)
cv2.createTrackbar("Upper Area", "Threshold 2", 0, 100000, f)

while (True):
    # read and resize images
    image = cv2.imread("colourshape.jpg")
    image = cv2.resize(image, (500, 500))

    # Object Segmentation
    ##########################################################
    # convert image to rgb
    converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # display in plot to read rgb values
    plt.imshow(converted_image)
    plt.show()

    # retrieve rgb boundaries values
    t1_R = cv2.getTrackbarPos("R", "Threshold 1")
    t1_G = cv2.getTrackbarPos("G", "Threshold 1")
    t1_B = cv2.getTrackbarPos("B", "Threshold 1")

    t2_R = cv2.getTrackbarPos("R", "Threshold 2")
    t2_G = cv2.getTrackbarPos("G", "Threshold 2")
    t2_B = cv2.getTrackbarPos("B", "Threshold 2")

    # rgb values
    threshold1 = (t1_R, t1_G, t1_B)
    threshold2 = (t2_R, t2_G, t2_B)

    # creating mask
    mask = cv2.inRange(converted_image, threshold1, threshold2)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Object Detection
    #######################################################
    # converting mask into grayscale for area calculation
    binary_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # find contours of the objects
    contours, _ = cv2.findContours(binary_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 1

    for contour in contours:
        x1, y1 = contour[0][0]

        # to get the area of each contour
        area = cv2.contourArea(contour)
        # retrieve boundaries of area
        lower_area_threshold = cv2.getTrackbarPos("Lower Area", "Threshold 1")
        upper_area_threshold = cv2.getTrackbarPos("Upper Area", "Threshold 2")
        if (lower_area_threshold < area < upper_area_threshold):
            print(area)
            # cv2.approxPolyDP() function to approximate the shape
            approx_contour = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            # using drawContour() function
            cv2.drawContours(image, [contour], 0, (0, 0, 255), 5)

            approx_contour = approx_contour[:, 0]

            left_upper_coordinate_x = approx_contour[0][0]
            left_upper_coordinate_y = approx_contour[0][1]
            # label objects
            cv2.putText(image, 'Object ' + str(i), (left_upper_coordinate_x, left_upper_coordinate_y),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 2)
            i += 1
    print(cv2.imwrite("Outcome.jpg", image))
    print(cv2.imwrite("Mask.jpg", mask))
    print(cv2.imwrite("Result.jpg", result))
    cv2.imshow('Original', image)
    cv2.imshow("binary", binary_result)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
