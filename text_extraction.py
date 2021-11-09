import numpy as np
import cv2
import math
from scipy import ndimage
import pytesseract
import os
import re
import datetime
from datetime import date


def using_tesseract():
    """
    default setting for tesseract
    """
    if (os.name) == "nt":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\shahr\anaconda3\envs\tesseract\Library\bin\tesseract.exe'
    return pytesseract.pytesseract.tesseract_cmd
pytesseract.pytesseract.tesseract_cmd = using_tesseract()



def img_preprossecing():
    img = cv2.imread('C:/Users/shahr/Downloads/BusinessCard.jpg')
    #cv2.imshow("Original", cv2.resize(img,(700,700)))
    #cv2.waitKey(0)
    #img=cv2.resize(img,(500,500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    #cv2.imshow("blur", blur)
    #cv2.waitKey(0)
    edges = cv2.Canny(blur, 100, 100, apertureSize=3)
    #edges=cv2.bitwise_not(blur)
    #cv2.imshow("edges", edges)
    #cv2.waitKey(0)
    #thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #cv2.imshow("thresh", thresh)
    #cv2.waitKey(0)

    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    #cv2.imshow("Detected lines", img)
    #key = cv2.waitKey(0)

    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img, median_angle)

    print(f"Angle is {median_angle:.04f}")
    #cv2.imshow('rotated.jpg', cv2.resize(img_rotated,(700,700)))
    #cv2.waitKey(0)
    return img_rotated,median_angle


def extract_data():
    """
    using Tesseract tries to read the date generated
    """
    conf = " -c tessedit_create_boxfile=1"
    text = str()
    img,median_angle=img_preprossecing()
    # print(date)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img=cv2.GaussianBlur(img, (7,7), 0)
    #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow('t', cv2.resize(img,(700,700)))
    cv2.waitKey(0)
    #img = cv2.Canny(img, 100, 100, apertureSize=3)
    #cv2.imshow('canny', cv2.resize(img,(700,700)))
    #cv2.waitKey(0)
    print(img.shape)
    imgh, imgw= img.shape
    boxes = pytesseract.image_to_boxes(img, config=conf)  # creates the boxes around each character
    # print(boxes)
    for b in boxes.splitlines():
        b = b.split()
        #print(b)
        x, y, h, w = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        img = cv2.rectangle(img, (x, imgh - y), (h, imgh - w), (0, 255, 0), 2)
        #cv2.putText(img, b[0], (x, imgh - y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        if str(b[0]) == 'M':
            pass
        else:
            text = text +' ' +str(b[0])
    #date=re.search(r'\d[4].\d[2].\d[2]', text)
    #date=datetime.datetime.strptime(date.group(), '%Y-%m-%d').date()
    print(text)
    #print(numbers)

    cv2.imshow('Resualt', cv2.resize(img,(700,700)))
    cv2.waitKey(0)
    return img, text
img,text=extract_data()

