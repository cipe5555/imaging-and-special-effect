import cv2
import pytesseract

# Set path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def f(x):
    return (x)

cv2.namedWindow("Text Type(0: Characters, 1: Words, 2: Digits)")
cv2.resizeWindow("Text Type(0: Characters, 1: Words, 2: Digits)", 300, 300)
cv2.createTrackbar("Type", "Text Type(0: Characters, 1: Words, 2: Digits)", 0, 2, f)
while (True):
    detection_type = cv2.getTrackbarPos("Type", "Text Type(0: Characters, 1: Words, 2: Digits)")

    # Read input image in BGR format
    img = cv2.imread("warpcolored.jpg")

    # Resize the image if it's too big
    max_width = 1000
    if img.shape[1] > max_width:
        scale_percent = max_width / img.shape[1] * 100
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Convert BGR image to RGB for compatibility with Tesseract OCR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ##print(pytesseract.image_to_string(img))
    if (detection_type == 0):
        ###Detecting Characters

        hImg, wImg, _ = img.shape
        boxes = pytesseract.image_to_boxes(img)
        for b in boxes.splitlines():
            # print(b)
            b = b.split(" ")
            print(b)
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 2)  # giving the colour RED and thickness 1
            cv2.putText(img, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 50, 255), 1)

    if (detection_type == 1):
        ## Detecting Words (after modification)
        try:
            hImg, wImg, _ = img.shape
            boxes = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            # Loop through each detected word and draw a rectangle around it
            for i in range(len(boxes['level'])):
                if int(boxes['conf'][i]) > 50:
                    x, y, width, height = boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i]
                    # Exclude small text regions
                    if width * height > 100:
                        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
                        cv2.putText(img, boxes['text'][i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        # Display the detected text in the terminal
                        print(f"Detected Text: {boxes['text'][i]}")
        except Exception as e:
            print(f"Error detecting words: {e}")
    if (detection_type == 2):
        ### Detecting Digit/number
        hImg, wImg, _ = img.shape
        config = "--oem 3 --psm 6 outputbase digits"
        boxes = pytesseract.image_to_data(img, config=config)
        for x, b in enumerate(boxes.splitlines()):
            if x != 0:
                b = b.split()
                print(b)
                if len(b) == 12:
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 2)
                    cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

    # Display the result
    cv2.imshow("Result", img)
    print(cv2.imwrite("TextDetection.jpg", img))

    if cv2.waitKey(200) & 0xFF == ord('q'):
        print("ffff")
        break
    cv2.waitKey(0)
cv2.destroyAllWindows()



