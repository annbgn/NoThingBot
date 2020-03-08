import time

import cv2
import numpy


def detect_edge(image):
    image = image.convert("RGB")
    cv_image = numpy.array(image)
    cv_image = cv_image[:, :, ::-1].copy()
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
    canny = cv2.Canny(cv_image, 30, 150)  # todo compute treshold or guess a good value
    t = time.localtime()
    timestamp = time.strftime("%b-%d-%Y_%H%M%S", t)
    cv2.imwrite(".\screenshots_\edge{}.jpg".format(timestamp), canny)
