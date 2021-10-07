import random
import cv2
import glob
import pytesseract
import os
import numpy as np
from date_generator import create_date

def using_tesseract():
    if (os.name) == "nt":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\shahr\anaconda3\envs\tesseract\Library\bin\tesseract.exe'
    return pytesseract.pytesseract.tesseract_cmd

# print(len(files))

# for file in files:
# rand=random.choice(file)
# img.append(rand)


# imageF=create_date()
# cv2.imshow('r', imageF)
# cv2.waitKey(0)
# print(type(dataaaa))

pytesseract.pytesseract.tesseract_cmd = using_tesseract()


def reading_date(date):
    conf = " -c tessedit_create_boxfile=1"
    text = str()
    d = date
    # print(date)
    d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
    imgh, imgw, _ = d.shape
    boxes = pytesseract.image_to_boxes(d, config=conf)
    # print(boxes)
    for b in boxes.splitlines():
        b = b.split(' ')
        # print(b)
        x, y, h, w = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        d = cv2.rectangle(d, (x, imgh - y), (h, imgh - w), (0, 255, 0), 2)
        cv2.putText(d, b[0], (x, imgh - y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        if str(b[0]) == '/':
            pass
        else:
            text = text + str(b[0])
    print(text)
    cv2.imshow('Resualt', d)
    cv2.waitKey(0)
    return d, text



def compare(name, text):
    if name == text:
        is_correct = 1  # if it's correct
    else:
        is_correct = 0  # if it's wrong
    return is_correct


def test():
    correct_dict = dict()
    wrong_dict = dict()
    count_total = 0
    for i in range(0, 10):
        date, date_name = create_date()
        d_text, extracted_text = reading_date(date)
        is_correct = compare(date_name, extracted_text)
        if not is_correct:
            wrong_dict.update({extracted_text: date})
        else:
            correct_dict.update({extracted_text: date})
            count_total = +1
    print('accuracy percentage:', count_total / 100)
    print('wrong:',len(wrong_dict))
    print('correct:', len(correct_dict))
    return d_text, wrong_dict, correct_dict, count_total


d, wrong, correct, count = test()
