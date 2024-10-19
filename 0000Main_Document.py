import numpy as np
import cv2
import subprocess

pathImage = "ipad.jpg"
heightImg = 570
widthImg = 545


# Combine all the image in a window
def stackImages(imgArray, scale, lables):
    Rows = len(imgArray)
    print("The Rows of the labels in a window: ", Rows)
    Columns = len(imgArray[0])
    print("The Columns of the labels in a window: ", Columns)

    # Find the height and width for output images
    height = imgArray[0][0].shape[0]
    width = imgArray[0][0].shape[1]

    # Check the available of multiple row label
    RowsAvailable = isinstance(imgArray[0], list)
    print("The array of images contains multiple rows of images: ", RowsAvailable)

    if RowsAvailable:
        for x in range(0, Rows, 1):
            for y in range(0, Columns, 1):
                # The shape of the current image is the same as the shape of the original image in the array
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), fx=scale, fy=scale, interpolation=None)

                # The shape of the current image is not the same as the shape of the original image
                else:
                    # Find the height and width for images
                    height = imgArray[0][0].shape[0]
                    width = imgArray[0][0].shape[1]
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (width, height), fx=scale, fy=scale, interpolation=None)

                # Check whether the current image is a grayscale image or a color image
                # If current image is 2D (len(imgArray[x][y].shape) == 2), it means that it is a grayscale image
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        # Create blank image
        imageBlank = np.zeros((height, width, 3), dtype=np.uint8)

        # Create the list to store horizontally concatenated images
        horizontal = [imageBlank] * Rows
        for x in range(0, Rows, 1):
            horizontal[x] = np.hstack(imgArray[x])
        vertical = np.vstack(horizontal)

    else:
        for x in range(0, Rows, 1):
            # The shape of the current image is the same as the shape of the original image in the array
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), fx=scale, fy=scale, interpolation=None)

            # The shape of the current image is not the same as the shape of the original image
            else:
                # Find the height and width for images
                height = imgArray[0][0].shape[0]
                width = imgArray[0][0].shape[1]
                imgArray[x] = cv2.resize(imgArray[x], (width, height), fx=scale, fy=scale, interpolation=None)

            # Check whether the current image is a grayscale image or a color image
            # If current image is 2D (len(imgArray[x][y].shape) == 2), it means that it is a grayscale image
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        horizontal = np.hstack(imgArray)
        vertical = horizontal

    # Put the labels names
    if len(lables) != 0:
        eachImgHeight = int(vertical.shape[0] / Rows)
        eachImgWidth = int(vertical.shape[1] / Columns)
        print("Each of the image height", eachImgHeight)
        print("Each of the image width:", eachImgWidth)

        for x1 in range(0, Rows, 1):
            for y1 in range(0, Columns, 1):
                cv2.putText(vertical, lables[x1][y1], (eachImgWidth * y1 + 5, eachImgHeight * x1 + 360),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 255, 255), thickness=2)
    return vertical


# Rearranges the points for form a rectangle.
def orderPoints(oldPoints):
    # Reshape old points array from 3D array to 2D array
    oldPoints = oldPoints.reshape((4, 2))
    # Create new points array
    newPoints = np.zeros((4, 1, 2), dtype=np.int32)

    # Compute the sum between the points
    addition = np.sum(oldPoints, axis=1)
    print("The sum of each row in the old points: ", addition)

    # The newPoints[0] = bottom-left point
    newPoints[0] = oldPoints[np.argmin(addition)]
    # The newPoints[2] = top-right point
    newPoints[2] = oldPoints[np.argmax(addition)]

    # Compute the difference between the points
    differences = np.diff(oldPoints, axis=1)
    print("The differences between consecutive points in each row of the points: \n", differences)

    # The newPoints[1] = bottom-right point
    newPoints[1] = oldPoints[np.argmin(differences)]
    # The newPoints[3] = top-left point
    newPoints[3] = oldPoints[np.argmax(differences)]

    print("The rectangle new points for the document contour: \n", newPoints)
    return newPoints


def empty():
    pass


# Trackbars
cv2.namedWindow("Canny and Area Trackbars")
cv2.resizeWindow("Canny and Area Trackbars", 300, 130)
cv2.createTrackbar("Threshold1", "Canny and Area Trackbars", 0, 255, empty)
cv2.createTrackbar("Threshold2", "Canny and Area Trackbars", 20, 255, empty)
cv2.createTrackbar("Area", "Canny and Area Trackbars", 5000, 30000, empty)


def getContour(contours):
    documentContour = np.array([])
    maxArea = 0
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        minArea = cv2.getTrackbarPos("Area", "Canny and Area Trackbars")
        if area > minArea:
            # Calculate the perimeter of the contour
            perimeter = cv2.arcLength(contour, True)
            # Epsilon =  0.02 times the perimeter  of the contour
            approximation = cv2.approxPolyDP(contour, epsilon=0.02 * perimeter, closed=True)
            if area > maxArea and len(approximation) == 4:
                documentContour = approximation
                maxArea = area
    return documentContour


def processImage(img):
    # Convert image to gray scale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    blurImg = cv2.GaussianBlur(grayscale, (5, 5), 0)
    # Get trackbar values for /thresholds
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Canny and Area Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Canny and Area Trackbars")
    # Canny edge detector
    thresholdImg = cv2.Canny(blurImg, Threshold1, Threshold2)
    kernel = np.ones((5, 5), np.uint8)
    # Morphological operations
    # Dilation
    DilationImg = cv2.dilate(thresholdImg, kernel, iterations=2)
    # Erosion
    thresholdImg = cv2.erode(DilationImg, kernel, iterations=1)
    return thresholdImg


def drawContours(thresholdImg):
    # Draw contours
    # Find all the contours
    contours, hierarchy = cv2.findContours(thresholdImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Copy image for display on the "Contour" label
    contoursImg = img.copy()
    # Draw all the detected contours
    cv2.drawContours(contoursImg, contours, -1, (0, 255, 0), 5, cv2.LINE_AA)
    return contours, contoursImg


def wrapDocument(img):
    # Prepare Points for wrap
    pts1 = np.float32(documentContour)
    pts2 = np.array([[0, 0], [widthImg, 0], [widthImg, heightImg], [0, heightImg]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    coloredDocument = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # Remove 20 pixels from each side
    coloredDocument = coloredDocument[20:coloredDocument.shape[0] - 20, 20:coloredDocument.shape[1] - 20]
    coloredDocument = cv2.resize(coloredDocument, (widthImg, heightImg))
    return coloredDocument


count = 0

while True:
    # Create blank image for testing debugging
    blankImg = np.zeros((heightImg, widthImg, 3), dtype=np.uint8)
    # Read image
    img = cv2.imread(pathImage)
    # Resize image
    img = cv2.resize(img, (widthImg, heightImg))
    # Processing image
    thresholdImg = processImage(img)
    # Draw contours
    contours, contoursImg = drawContours(thresholdImg)

    # Find the documentContour contour
    documentContour = getContour(contours)
    if documentContour.size != 0:
        documentContour = orderPoints(documentContour)
        # Copy image for display on the "documentContour Contour" label
        imgBigContour = img.copy()
        # Draw the documentContour contour
        cv2.drawContours(imgBigContour, [documentContour], -1, (0, 255, 0), 8)
        # Wrap Document
        coloredDocument = wrapDocument(img)
        grayDocument = cv2.cvtColor(coloredDocument, cv2.COLOR_BGR2GRAY)

        # Image Array for Display
        arrayOfImage = ([img, thresholdImg, contoursImg],
                        [imgBigContour, coloredDocument, grayDocument])

        ### Square&Rectangles ###
        # Filter contours based on size
        lengthWidthImg = imgBigContour.copy()
        min_contour_area = 5000  # Minimum contour area threshold
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        # Remove the largest contour (likely the window)
        # filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[1:]
        # Loop over filtered contours
        for cnt in filtered_contours:
            print(filtered_contours)
            lengthWidthImg = imgBigContour.copy()
            # Find minimum area rectangle
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            # Calculate length and width in centimeters
            w, h = rect[1]
            pixel_to_cm_ratio = 0.0264583333  # Define your own pixel to cm ratio
            length = max(w, h) * pixel_to_cm_ratio
            width = min(w, h) * pixel_to_cm_ratio
            # Add text for length and width in centimeters
            text_x, text_y = box[1]
            cv2.putText(lengthWidthImg, "W: {:.2f} cm".format(length), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(lengthWidthImg, "L: {:.2f} cm".format(width), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

    else:
        arrayOfImage = ([img, thresholdImg, contoursImg],
                        [blankImg, blankImg, blankImg])

    # LABELS FOR DISPLAY
    labels = [["Original", "Threshold", "Contours"],
              ["Document Contour", "Coloured Document", "Gray Document"]]

    stackedImage = stackImages(arrayOfImage, 0.65, labels)

    cv2.imshow("Document Scanner", stackedImage)
    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 'q' to exit
        cv2.imshow("Detect document length and width", lengthWidthImg)
        print(cv2.imwrite("LengthWidth.jpg", lengthWidthImg))
        print(cv2.imwrite("warpcolored.jpg", coloredDocument))
        print(cv2.imwrite("stackedImage.jpg", stackedImage))
        print("saved")
        cv2.waitKey(0)
        count += 1
        break

cv2.destroyAllWindows()
subprocess.call(['python', '0000TextDetection.py'])
subprocess.call(['python', '0000ShapesnAreas.py'])
subprocess.call(['python', '0000Colours.py'])
