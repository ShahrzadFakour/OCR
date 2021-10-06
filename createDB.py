import random
import numpy as np
import cv2
import glob


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
    for i in range(0, 10):
        img = [random.choice(file) for file in files]
    return img


# print(img)
# for im in img:
# imgs.append(cv2.imread(im))
img = choose_random_images()


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
    for d in divider:
        cv2.imshow('d',d)
        cv2.waitKey(0)
    return divider


imgs = read_images()


def create_day():
    """
    Return an image from 1 to 31 and its name
    :return:
    """
    day = []
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
            dName = str(i) + str(j)

    return day, dName


def create_month():
    month = []
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
            mName = str(m) + str(j)

    return month, mName


def create_year():
    year = []
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
                yName = '2' + str(j) + str(k) + str(x)
    return year, yName


day, dName = create_day()
month, mName = create_month()
year, yName = create_year()
div = read_divider()


def create_date():
    date = []
    for i in range(1000):
        white = [255, 255, 255]
        d = random.choice(day)
        m = random.choice(month)
        y = random.choice(year)
        divider = random.choice(div)
        dd = cv2.hconcat([d, divider])
        mm = cv2.hconcat([m, divider])
        dm = cv2.hconcat([dd, mm])
        fullDate = cv2.hconcat([dm, y])
        image = cv2.copyMakeBorder(fullDate, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=white)
        cv2.imshow('f', fullDate)
        cv2.waitKey(0)
        date.append(image)
    print(len(date))
    return date


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
data = create_date()
# imageF=create_date()
# cv2.imshow('r', imageF)
# cv2.waitKey(0)
#print(type(dataaaa))

