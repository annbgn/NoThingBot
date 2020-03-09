import time

import cv2
import numpy


def detect_edge(image):
    image = image.convert("RGB")
    cv_image = numpy.array(image)
    cv_image = cv_image[:, :, ::-1].copy()
    cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv_image_gray = cv2.GaussianBlur(cv_image_gray, (5, 5), 0)

    sigma15 = 0.15
    sigma20 = 0.20
    sigma25 = 0.25

    # compute the median of the single channel pixel intensities
    v = numpy.median(image)
    # apply automatic Canny edge detection using the computed median
    lower15 = int(max(0, (1.0 - sigma15) * v))
    upper15 = int(min(255, (1.0 + sigma15) * v))

    lower20 = int(max(0, (1.0 - sigma20) * v))
    upper20 = int(min(255, (1.0 + sigma20) * v))

    lower25 = int(max(0, (1.0 - sigma25) * v))
    upper25 = int(min(255, (1.0 + sigma25) * v))

    canny15 = cv2.Canny(cv_image_gray, lower15, upper15)
    canny20 = cv2.Canny(cv_image_gray, lower20, upper20)
    canny25 = cv2.Canny(cv_image_gray, lower25, upper25)
    t = time.localtime()
    timestamp = time.strftime("%b-%d-%Y_%H%M%S", t)
    # hist = get_hist(cv_image_gray)
    cv2.imwrite(
        ".\screenshots_\\full{}.jpg".format(timestamp),
        numpy.hstack(
            [
                cv_image,
                cv2.cvtColor(canny15, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(canny20, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(canny25, cv2.COLOR_GRAY2BGR),
            ]
        ),
    )


def get_hist(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist
