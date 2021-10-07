import random
import cv2
import glob
import pytesseract
import os
import numpy as np


def using_tesseract():
    if (os.name) == "nt":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\shahr\anaconda3\envs\tesseract\Library\bin\tesseract.exe'
    return pytesseract.pytesseract.tesseract_cmd


def find_digit_path():
    path = r'C:\Users\shahr\Downloads\archive\EXP_10'
    files = []
    for i in range(0, 10):
        digit = glob.glob(path + '/' + str(i) + '/*.png', recursive=True)
        files.append(digit)
    return files


# print(len(files))

# for file in files:
# rand=random.choice(file)
# img.append(rand)
files = find_digit_path()


def choose_random_images():
    img=[]
    for i in range(0, 10):
        number=[i for i in range(0,10)]
        img = [random.choice(file) for file in files]

    return img,number



# for im in img:
# imgs.append(cv2.imread(im))
img,number = choose_random_images()
#print(name)


def read_images():
    imgs = [cv2.imread(i) for i in img]
    return imgs


def read_divider():
    slash = r'C:\Users\shahr\Downloads\slash.png'
    space = r'C:\Users\shahr\Downloads\space.png'
    div = []
    div.append(slash)
    div.append(space)
    divider = [cv2.imread(d) for d in div]
    # for d in divider:
    # cv2.imshow('d', d)
    # cv2.waitKey(0)
    return divider


imgs = read_images()


def create_day():
    """
    Return an image from 1 to 31 and its name
    :return:
    """
    day = []
    name = []
    dName = dict()
    for i in range(0, 4):
        day1 = imgs[i]
        if i == 0:
            start = 1
            end = 10
        elif 0 < i < 3:
            start = 1
            end = 10
        elif i == 3:
            start = 0
            end = 2
        for j in range(start, end):
            day2 = imgs[j]
            fullDay = cv2.hconcat([day1, day2])
            day.append(fullDay)
            # print(fullDay)
            # cv2.imshow('r', fullDay)
            # cv2.waitKey(0)
            n = str(i) + str(j)
            name.append(n)
            dName = dict(zip(name, day))
    # print(dName)

    return day, dName


def create_month():
    month = []
    name = []
    mName = []
    for m in range(0, 2):
        month1 = imgs[m]

        if m == 0:
            start = 1
            end = 10
        else:
            start = 0
            end = 3
        for j in range(start, end):
            month2 = imgs[j]
            fullMonth = cv2.hconcat([month1, month2])
            # cv2.imshow('r',fullMonth)
            # cv2.waitKey(0)
            month.append(fullMonth)
            name.append(str(m) + str(j))
            mName = dict(zip(name, month))
    # print(mName)

    return month, mName


def create_year():
    year = []
    name = []
    # create year
    y1 = imgs[2]
    for j in range(0, 10):
        y2 = imgs[j]
        y12 = cv2.hconcat([y1, y2])
        # cv2.imshow('r',full)
        # cv2.waitKey(0)
        # day.append(y12)
        for k in range(2, 9):
            y3 = imgs[k]
            for x in range(0, 9):
                y4 = imgs[x]
                y34 = cv2.hconcat([y3, y4])
                fullYear = cv2.hconcat([y12, y34])
                # cv2.imshow('r', fullYear)
                # cv2.waitKey(0)
                year.append((fullYear))
                name.append('2' + str(j) + str(k) + str(x))
                yName = dict(zip(name, year))
    # print(yName)
    return year, yName


day, dName = create_day()
month, mName = create_month()
year, yName = create_year()
div = read_divider()


def create_date():
    date = []
    name = []
    for i in range(0, 5):
        white = [255, 255, 255]
        d = random.choice(list(dName.items()))
        dN = d[0]  # name
        dV = d[1]  # day image

        m = random.choice(list(mName.items()))
        mN = m[0]
        mV = m[1]
        y = random.choice(list(yName.items()))
        yN = y[0]
        yV = y[1]

        divider = random.choice(div)

        dd = cv2.hconcat([dV, divider])
        mm = cv2.hconcat([mV, divider])
        dm = cv2.hconcat([dd, mm])
        fullDate = cv2.hconcat([dm, yV])
        image = cv2.copyMakeBorder(fullDate, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=white)
        # cv2.imshow('f', fullDate)
        # cv2.waitKey(0)
        date.append(image)
        name = (dN + mN + yN)
        print(name)
    # print(dateName)

    return date, name


'''''
x = []
day, dName = create_day()
month, mName = create_month()
year, yName = create_year()
separator = '-'
name = f'{dName}{separator}{mName}{separator}{yName}'
print([day, month,year])
#print(name)
'''''
data, name = create_date()
# imageF=create_date()
# cv2.imshow('r', imageF)
# cv2.waitKey(0)
# print(type(dataaaa))

pytesseract.pytesseract.tesseract_cmd = using_tesseract()


def reading_dates():
    conf = " -c tessedit_create_boxfile=1"
    date = []
    for d in data:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        imgh, imgw, _ = d.shape
        boxes = pytesseract.image_to_boxes(d, config=conf)
        print(boxes)
        for b in boxes.splitlines():
            b = b.split(' ')
            #print(b)
            x, y, h, w = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            img = cv2.rectangle(d, (x, imgh - y), (h, imgh - w), (0, 255, 0), 2)
            cv2.putText(d, b[0], (x, imgh - y + 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        cv2.imshow('Resualt', d)
        cv2.waitKey(0)
    return d


read = reading_dates()
