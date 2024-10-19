# import the necessary packages
import numpy as np
import argparse
import cv2
import pandas as pd

img = cv2.imread("warpcolored.jpg")

# declaring global variables (are used later on)
clicked = False
r = g = b = xpos = ypos = 0

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# function to calculate minimum distance from all colors and get the most matching color
def getColorName(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


# function to get x,y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)


cv2.namedWindow('Colour Detection')
cv2.setMouseCallback('Colour Detection', draw_function)

while (1):

    cv2.imshow("Colour Detection", img)
    new_img = img.copy()
    if (clicked):

        # cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle
        cv2.rectangle(new_img, (0, 0), (458, 30), (b, g, r), -1)

        # Creating text string to display( Color name and RGB values )
        text = getColorName(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)


        # cv2.putText(new_img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(new_img, text, (12, 21), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(new_img, text, (12, 21), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        # For very light colours we will display text in black colour
        if (r + g + b >= 600):
            cv2.putText(new_img, text, (12, 21), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(new_img, text, (12, 21), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        clicked = False
        cv2.imshow("Colour Detection", new_img)
        print(cv2.imwrite("Color Detected.jpg", new_img))
        cv2.waitKey(3000)

    # Break the loop when user hits 'esc' key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()

