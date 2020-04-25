import time

import cv2
import numpy


def detect_edge(image):
    image = image.convert("RGB")
    cv_image = numpy.array(image)
    cv_image = cv_image[:, :, ::-1].copy()
    cv_image_gray_wo_contrast = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv_image_gray = cv2.GaussianBlur(cv_image_gray_wo_contrast, (5, 5), 0)

    if is_low_contrast(hist=get_hist(cv_image_gray)):
        cv_image_gray = increase_contrast(cv_image_gray)

    # compute the median of the single channel pixel intensities
    v = numpy.median(image)  # v = numpy.mean(image)

    # принцип: яркое - уменьшать сигму, темное - увеличить сигму
    sigma15 = 0.10
    sigma20 = 0.30
    sigma25 = 0.50

    if v > 255 / 2:
        sigma15 = sigma15 / 2
        sigma20 = sigma20 / 2
        sigma25 = sigma25 / 2
    else:
        sigma15 = sigma15 * 2
        sigma20 = sigma20 * 2
        sigma25 = sigma25 * 2

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
                cv2.cvtColor(cv_image_gray_wo_contrast, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(cv_image_gray, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(canny15, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(canny20, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(canny25, cv2.COLOR_GRAY2BGR),
            ]
        ),
    )


def get_hist(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist


def is_low_contrast(hist):
    # validation: hist must be black&white image histogram
    assert hist.shape[0] == 256
    assert hist.shape[1] == 1

    mean = hist.mean()  # for gray images hist mean = height * width / 256
    fluctuation = 400  # const,  на которую может отличаться количество пикселей от среднего. ее надо подобрать.в идеале надо чтобы она составляла какой-то процент, напр 10%, от mean
    result = list(
        filter(
            lambda x: bool(
                x[0] not in range(int(mean - fluctuation), int(mean + fluctuation) + 1)
            ),
            hist,
        )
    )  # тут может оказаться медленно, может надо перерписать обычным фориком или setcomp'ом, но тогда надо позасекать время выполнения кек
    return bool(result)


def increase_contrast(image):
    """image is b&w"""
    new_image = numpy.zeros(image.shape, image.dtype)
    alpha = 1.5  # contrast. must be >1
    beta = 0  # brightness. i'm not using it now, but it might be useful for evaluating sigmas for threshold
    for i in range(image.shape[0]):
        new_image[i] = numpy.clip(alpha * image[i] + beta, 0, 255)
    return new_image
