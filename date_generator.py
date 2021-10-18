import glob
import cv2
import random


def find_digit_path():
    """
    this method puts all images of directory in a list
    """
    path = r'C:\Users\shahr\Downloads\myData'
    files = []
    for i in range(0, 10):
        digit = glob.glob(path + '/' + str(i) + '/*.png', recursive=True)
        files.append(digit)
    return files


# List of list containing for each index, a list of images of the digits equals to the index name
FILES = find_digit_path()

#print(FILES[1])
def read_divider():
    """
    reads our dividers and returns the list of dividers
    """
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


def create_day():
    """
    Returns an image from 1 to 31 and its name

    """
    day = []
    name = []
    dName = dict()
    imgs = [cv2.imread(random.choice(file)) for file in FILES]
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
            dName = dict(zip(name, day))    # create a dictionary in which names are keys and the images are values
    # print(dName)

    return day, dName

day,dName=create_day()
print(dName)
def create_month():
    """
    Returns an image of numbers from 1 to 12 for each month with its name
    """
    month = []
    name = []
    mName = []
    imgs = [cv2.imread(random.choice(file)) for file in FILES]
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
            mName = dict(zip(name, month))  # create a dictionary in which names are keys and the images are values
    # print(mName)

    return month, mName


def create_year():
    """
    Returns a list of years from 2020
    """
    year = []
    name = []
    # create year
    imgs = [cv2.imread(random.choice(file)) for file in FILES]
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
                yName = dict(zip(name, year))  # create a dictionary in which names are keys and the images are values
    # print(yName)
    return year, yName


def create_date():
    """
     using output of previous methods, returns an image that demonstrates a complete date
    """
    day, dName = create_day()
    month, mName = create_month()
    year, yName = create_year()
    div = read_divider()

    white = [255, 255, 255]
    d = random.choice(list(dName.items()))
    dN = d[0]  # name of day
    dV = d[1]  # image of day
    dV=cv2.resize(dV,(64,32))

    m = random.choice(list(mName.items()))
    mN = m[0]  # name of month
    mV = m[1]  # image of month
    mV=cv2.resize(mV,(64,32))
    y = random.choice(list(yName.items()))
    yN = y[0]  # name of year
    yV = y[1]  # image of year
    yV= cv2.resize(yV,(128,32))

    divider = random.choice(div)

    dd = cv2.hconcat([dV, divider])
    mm = cv2.hconcat([mV, divider])
    dm = cv2.hconcat([dd, mm])
    fullDate = cv2.hconcat([dm, yV])
    date = cv2.copyMakeBorder(fullDate, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=white)  # create a board for date
    cv2.imshow('fullDay', date)
    cv2.waitKey(0)
    name = (dN + mN + yN)  # create the name of date that contains names of day,month and year
    print(name)
    # print(dateName)

    return date, name
