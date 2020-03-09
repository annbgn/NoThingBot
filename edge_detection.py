from matplotlib import pyplot as plt
import time

import cv2
import numpy


def detect_edge(image, sigma=0.15):
    image = image.convert("RGB")
    cv_image = numpy.array(image)
    cv_image = cv_image[:, :, ::-1].copy()
    cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv_image_gray = cv2.GaussianBlur(cv_image_gray, (5, 5), 0)

    # compute the median of the single channel pixel intensities
    v = numpy.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    canny = cv2.Canny(cv_image_gray, lower, upper)
    t = time.localtime()
    timestamp = time.strftime("%b-%d-%Y_%H%M%S", t)
    hist = show_hist(cv_image_gray, lower, upper, timestamp)
    cv2.imwrite(
        ".\screenshots_\\full{}.jpg".format(timestamp),
        numpy.hstack([cv_image, cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)]),
    )  #  , cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR)]))


def show_hist(image, low_boundary, high_boundary, timestamp):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

    plt.figure()

    plt.title("Grayscale Histogram")

    plt.xlabel("Bins")

    plt.ylabel("# of Pixels")

    plt.plot(hist)

    plt.xlim([0, 256])

    plt.savefig(".\screenshots_\hist{}.jpg".format(timestamp))

    plt.close()
