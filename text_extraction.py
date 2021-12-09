import numpy as np
import cv2
import math
from scipy import ndimage
import pytesseract
import os


def using_tesseract():
    """
    default setting for tesseract
    """
    if (os.name) == "nt":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\shahr\anaconda3\envs\tesseract\Library\bin\tesseract.exe'
    return pytesseract.pytesseract.tesseract_cmd

def load_model():
    model = load_model('finalized_model_10Epoche.h5')
    return model

def capture():
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        #success, org_img = cap.read()
        #img=np.asarray(org_img)
        img=cv2.imread('C:/Users/shahr/Downloads/BusinessCard.jpg')
        #print(frame)
        #img=preProcessing(img)
        cv2.imshow('frame',img)
        #img = cv2.resize(img, (32, 32))
        #img=img.reshape(1,32,32,3)
        #print(img.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return img

def rotate_image():
    #img = cv2.imread('C:/Users/shahr/Downloads/n.png')
    #cv2.imshow("Original", cv2.resize(img,(700,700)))
    #cv2.waitKey(0)
    img=capture()
    img=cv2.resize(img,(500,500))
    #img=capture()
    #print(img.dtype)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", cv2.resize(gray,(700,700)))
    cv2.waitKey(0)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    cv2.imshow("blur", cv2.resize(blur,(700,700)))
    cv2.waitKey(0)
    edges = cv2.Canny(blur, 100, 100, apertureSize=3)
    #edges=cv2.bitwise_not(blur)
    cv2.imshow("edges", cv2.resize(edges,(700,700)))
    cv2.waitKey(0)
    #thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #cv2.imshow("threshold", thresh)
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

def img_preprossecing():
    img, median_angle=rotate_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #img=cv2.GaussianBlur(img, (7,7), 0)
    #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow('processed image', cv2.resize(img, (700, 700)))
    cv2.waitKey(0)
    # img = cv2.Canny(img, 100, 100, apertureSize=3)
    # cv2.imshow('canny', cv2.resize(img,(700,700)))
    # cv2.waitKey(0)
    print(img.shape)
    return img


def extract_data():
    """
    using Tesseract tries to read the date generated
    """
    pytesseract.pytesseract.tesseract_cmd = using_tesseract()
    conf = " -c tessedit_create_boxfile=1 "
    text = str()
    img=img_preprossecing()
    # print(date)

    imgh, imgw= img.shape
    boxes = pytesseract.image_to_data(img, config=conf)  # creates the boxes around each character
    #print(boxes)
    for x, b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            #print(b)
            if len(b) == 12:
                print(b)
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                #print(x, y, w, h)
                img = cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
                #cv2.putText(img, b[11], (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 0), 1)
                text=text+' '+ str(b[11])
    #date=re.search(r'\d[4].\d[2].\d[2]', text)
    #date=datetime.datetime.strptime(date.group(), '%Y-%m-%d').date()
    print(text)
    #print(numbers)

    cv2.imshow('Resualt', cv2.resize(img,(700,700)))
    cv2.waitKey(0)
    return img, text
img,text=extract_data()
