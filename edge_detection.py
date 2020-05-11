from __future__ import division
import itertools
import copy
import logging
import time
from math import sqrt, acos
from pywinauto.keyboard import send_keys
import winsound

import cv2
import numpy

logging.basicConfig(
    filename="./app.log", format="%(asctime)s - %(levelname)s - %(message)s"
)


def detect_edge(image):

    t = time.localtime()
    timestamp = time.strftime("%b-%d-%Y_%H%M%S", t)

    image = image.convert("RGB")
    cv_image = numpy.array(image)
    cv_image = cv_image[:, :, ::-1].copy()
    cv_image_gray_wo_contrast = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # cv_image_gray = cv2.GaussianBlur(cv_image_gray_wo_contrast, (5, 5), 0)
    cv_image_gray = cv_image_gray_wo_contrast

    if is_low_contrast(hist=get_hist(cv_image_gray)):
        cv_image_gray = increase_contrast(cv_image_gray)

    # compute the median of the single channel pixel intensities
    v = numpy.median(image)  # v = numpy.mean(image)

    # принцип: яркое - уменьшать сигму, темное - увеличить сигму
    sigma1 = 0.10
    sigma2 = 0.30

    if (
        v > 255 / 2
    ):  # todo: after removing second canny, upper/lower and sigma, replace with one-line if-else
        sigma1 = sigma1 / 2
        sigma2 = sigma2 / 2
    else:
        sigma1 = sigma1 * 2
        sigma2 = sigma2 * 2

    # apply automatic Canny edge detection using the computed median
    lower1 = int(max(0, (1.0 - sigma1) * v))
    upper1 = int(min(255, (1.0 + sigma1) * v))

    lower2 = int(max(0, (1.0 - sigma2) * v))
    upper2 = int(min(255, (1.0 + sigma2) * v))

    canny1 = cv2.Canny(cv_image_gray, lower1, upper1)

    canny_wo_contrast = cv2.Canny(cv_image_gray, lower2, upper2)

    with_lines = draw_lines(canny1, timestamp=timestamp)

    corners_from_lines = find_corners(with_lines)
    corners_from_original = find_corners(cv_image_gray)


    cv2.imwrite(
        ".\screenshots_\\full{}.jpg".format(timestamp),
        numpy.vstack(
            (
                # numpy.hstack(
                #     [
                #         cv_image,  # original
                #         cv2.cvtColor(
                #             cv_image_gray_wo_contrast, cv2.COLOR_GRAY2BGR
                #         ),  # b&w, w/o improved contrast
                #         cv2.cvtColor(
                #             cv_image_gray, cv2.COLOR_GRAY2BGR
                #         ),  # improved contrast
                #         # cv2.cvtColor(corners_from_original, cv2.COLOR_GRAY2BGR), # corners
                #     ]
                # ),
                numpy.hstack(
                    [
                        # cv2.cvtColor(canny_wo_contrast, cv2.COLOR_GRAY2BGR),  # contours w/o contrast
                        cv2.cvtColor(
                            canny1, cv2.COLOR_GRAY2BGR
                        ),  # contours with threshold (empirically ok    )
                        cv2.cvtColor(with_lines, cv2.COLOR_GRAY2BGR),  # found lines
                        cv2.cvtColor(numpy.zeros(with_lines.shape, with_lines.dtype), cv2.COLOR_GRAY2BGR),  # only horisontal lines
                        # cv2.cvtColor(corners_from_lines, cv2.COLOR_GRAY2BGR),  # corners
                        # cv2.cvtColor(numpy.minimum(corners_from_original, corners_from_lines), cv2.COLOR_GRAY2BGR),  # AND
                    ]
                ),
                # numpy.hstack(
                #     [
                #         cv2.cvtColor(
                #             corners_from_original, cv2.COLOR_GRAY2BGR
                #         ),  # corners from bw image
                #         cv2.cvtColor(corners_from_lines, cv2.COLOR_GRAY2BGR),  # corners
                #         cv2.cvtColor(
                #             numpy.minimum(corners_from_original, corners_from_lines),
                #             cv2.COLOR_GRAY2BGR,
                #         ),  # AND
                #     ]
                # ),
                # numpy.hstack(
                #     [
                #         cv2.cvtColor(
                #             intersections, cv2.COLOR_GRAY2BGR
                #         ),  # intersections of horizontal lines
                #         cv2.cvtColor(with_lines, cv2.COLOR_GRAY2BGR),  # tmp
                #         cv2.cvtColor(with_lines, cv2.COLOR_GRAY2BGR),  # tmp
                #         # cv2.cvtColor(cv2.UMat(only_corners),cv2.COLOR_GRAY2BGR),  # only crossing pair of lines
                #     ]
                # ),
            )
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
    res = bool(result)
    logging.info(res)
    return res


def increase_contrast(image):
    """
    :param image: b&w image
    :return: image with higher contrast
    """
    new_image = numpy.zeros(image.shape, image.dtype)
    mean = image.mean()
    alpha = 1.5 if mean < 128 else 1.2  # contrast. must be >1
    logging.info("alpha = {}, mean = {}".format(alpha, mean))
    beta = 0  # brightness. i'm not using it now, but it might be useful for evaluating sigmas for threshold
    for i in range(image.shape[0]):
        new_image[i] = numpy.clip(alpha * image[i] + beta, 0, 255)
    return new_image


def draw_lines(canny, exclude=False, timestamp = None):
    """
    :param canny: image with edges already detected. Must be GRAY. # todo: assert GRAY
    :param exclude: remove lines which are possibly not connected with road
    
    :return: image with only lines
    """
    canny_copy = copy.deepcopy(canny)
    lines = cv2.HoughLinesP(canny_copy, 1, numpy.pi / 180, 100, minLineLength=20, maxLineGap=70)  # dilate to increase probability of intersection?/
    image_with_lines = numpy.zeros(canny.shape, canny.dtype)

    try:
        lines[0]
        # lines[0][3]
    except TypeError as exc:  # todo: why?
        logging.exception(exc)
        return image_with_lines, image_with_lines  # both are just zeros
    else:
        for i in lines:
            x1, y1, x2, y2 = i[0]
            if exclude:
                # todo: filter lines: remove everything above horizon, lines with impossible angle/tg, etc
                pass
            image_with_lines = cv2.line(
                image_with_lines, (x1, y1), (x2, y2), (255, 255, 255), 1
            )
        #
        # image_with_horizontals = numpy.zeros(canny.shape, canny.dtype)
        # image_wo_horizontals = numpy.zeros(canny.shape, canny.dtype)
        horizontals = []
        not_horizontals_on_the_right = []
        not_horizontals_on_the_left = []
        for i in lines:
            x1, y1, x2, y2 = i[0]
            if exclude:
                # todo: filter lines: remove everything above horizon, lines with impossible angle/tg, etc
                pass
            if y1 == y2:
                horizontals.append(i)
                # image_with_horizontals = cv2.line(
                #     image_with_horizontals, (x1 , y1), (x2, y2), (255, 255, 255), 1
                # )
            else:
                if x1 and x2 < 1919/2:
                    not_horizontals_on_the_left.append(i)
                else:
                    not_horizontals_on_the_right.append(i)
                # not_horizontals.append(i)
                # image_wo_horizontals = cv2.line(
                #     image_wo_horizontals, (x1, y1), (x2, y2), (255, 255, 255), 1
                # )
        # image_intersection_points = image_with_horizontals/2 + image_wo_horizontals/2
        # image_intersection_points[image_intersection_points < 255] = 0
        # image_intersection_points = image_intersection_points.astype(
        #     canny.dtype
        # )  # 'uint8'
        '''
        dilated = numpy.zeros(canny.shape, canny.dtype)
        dilated = cv2.dilate(image_intersection_points, None, dst = dilated, iterations=2)
        intersection_points = numpy.where(dilated == [255])
        intersection_points = zip(intersection_points[0], intersection_points[1])
        matching_lines = []
        for x, y in intersection_points:
            for hor in horizontals:
                for nhor in not_horizontals:
                    print('\n')
                    if is_crossing(x, y, hor, nhor):
                        matching_lines += ((x, y), hor, nhor)
        # print(matching_lines)
        only_corners = numpy.zeros(canny.shape, canny.dtype)
        for i in matching_lines:
            hx1, hy1, hx2, hy2 = i[1][0]
            only_corners = cv2.line(
                only_corners, (hx1, hy1), (hx2, hy2), (255, 255, 255), 1
            )
            nhx1, nhy1, nhx2, nhy2 = i[2][0]
            only_corners = cv2.line(
                only_corners, (nhx1, nhy1), (nhx2, nhy2), (255, 255, 255), 1
            )
        '''
        horizontals = list(filter(lambda  x: x[0][1] > (755 - 107 )/2, horizontals))  # we only need horisontals which are under horison
        reasons_to_turn_right, _ = reasons_to_turn(horizontals, not_horizontals_on_the_right, True, canny.shape, canny.dtype)
        reasons_to_turn_left, image_reasons = reasons_to_turn(horizontals, not_horizontals_on_the_left, False, canny.shape, canny.dtype)
        print(reasons_to_turn_left)
        print(reasons_to_turn_right)
        print('\n')
        cv2.imwrite('./screenshots_/ihatekursach_{}.jpg'.format(timestamp), image_reasons)
        if reasons_to_turn_right:
            turn_right()
            winsound.Beep(400, 100)
            winsound.Beep(600, 100)
        elif reasons_to_turn_left:
            turn_left()
            winsound.Beep(400, 100)
            winsound.Beep(400, 100)
            winsound.Beep(400, 100)
        '''
                # угол
                # cos (theta) = (A1A2 + B1B2)/(sqrt(A1^2 + B1^2)*sqrt(A2^2 + B2^2))
                try:
                    cos_angle = (horA*nhorA + horB*nhorB)/(sqrt(horA**2 + horB**2)*sqrt(nhorA**2 + nhorB**2))
                    angle = acos(cos_angle)
                except ZeroDivisionError:
                    continue
                else:
                    if angle > 0:
                        rea
                '''

                # cv2.imwrite('./tmp/debug_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(hx1, hy1, hx2, hy2, nhx1, nhy1, nhx2, nhy2 ), numpy.hstack((tmp1, tmp2)))
                # todo оптимизация: тут убрать все оставшиеся itertuls.product, содержащие hor или nhor
                # product = set(filter(lambda x: x[0] != hor and x[1] != nhor, product))

        #
        # new_image = numpy.zeros(canny.shape, canny.dtype)
        # drawn_intersections = cv2.dilate(
        #     image_intersection_points, None, dst=new_image, iterations=2
        # )

        return image_with_lines#, image_with_horizontals# , drawn_intersections


def find_corners(image):
    """
    :param image: b&w image
    :return:
    """
    gray = copy.deepcopy(image)
    gray = numpy.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    corners_marked = mark_corners(gray, dst)

    return corners_marked


def mark_corners(gray, dst):
    new_image = numpy.zeros(gray.shape, gray.dtype)
    new_image[dst > 0.01 * dst.max()] = 255
    cv2.dilate(new_image, None, dst=new_image, iterations=2)  # makes dots bolder
    return new_image


def is_crossing(x, y, line1, line2):
    # todo think if check for line1 == line2 or line1 is parallel to line2 is needed
    # solution: no, because this func is used only for horizontal and non-horizontal lines
    if is_in_segement(x, y, line1) and is_in_segement(x, y, line2):
        return True
    return False


def is_in_segement(x, y, line):
    """
    checks if a point (x,y) belongs to segment
    compares if the distance between start of segment to (x,y) + the distance between (x,y) and end of segment = full segment length
    """
    x1, y1, x2, y2 = line[0]
    print( hypotenuse(x1, y1, x, y) , ' + ', hypotenuse(x2, y2, x, y), ' == ', hypotenuse(x1, y1, x2, y2))
    return hypotenuse(x1, y1, x, y) + hypotenuse(x2, y2, x, y) == hypotenuse(x1, y1, x2, y2)


def hypotenuse(x1, y1, x2, y2):
    print(x1, y1, x2, y2)
    res = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    print(res)
    return res


def turn_right():
    send_keys("a")


def turn_left():
    send_keys("d")


def reasons_to_turn(horizontals, not_horisontals, right: bool, shape, dtype ):
    image_reasons = numpy.zeros(shape, dtype)
    reasons = 0
    product = itertools.product(horizontals, not_horisontals)
    for hor, nhor in product:
        tmp1 = numpy.zeros(shape, dtype)
        tmp2 = numpy.zeros(shape, dtype)
        hx1, hy1, hx2, hy2 = hor[0]
        nhx1, nhy1, nhx2, nhy2 = nhor[0]
        cv2.line(tmp1, (hx1, hy1), (hx2, hy2), 255, 5)
        cv2.line(tmp2, (nhx1, nhy1), (nhx2, nhy2), (255, 255, 255), 5)
        tmp = tmp1 / 2 + tmp2 / 2
        tmp[tmp < 255] = 0
        tmp = tmp.astype(
            dtype
        )  # 'uint8'
        if tmp.any():
            pass
            # общеее уравнение прямой по 2 точкам: (y - y1)/(y2 -y1) = (x - x1)/(x2 - x1)
            # или в общем виде: (y1 - y2) x + (x2 - x1) y + (x1y2 - x2y1) = 0
            horA = hy1 - hy2
            horB = hx2 - hx1
            horC = hx1 * hy2 - hx2 * hy1
            nhorA = nhy1 - nhy2
            nhorB = nhx2 - nhx1
            nhorC = nhx1 * nhy2 - nhx2 * nhy1
            # в виде y = kx + b нас интересует только знак k, чтобы определить, в какую сторону поворачивать, причем k интересует только у негоризонтальной прямой, т.к у горизонтальной k = 0
            # k = -A/B
            if nhorB == 0:
                # we don't vertical lines. morover it's ZeroDivision
                continue
            else:
                K = - (nhorA / nhorB)
                if right and K > 0 :
                    reasons += 1
                    image_reasons = cv2.line(image_reasons, (hx1, hy1), (hx2, hy2), 255, 2, cv2.LINE_AA )
                    image_reasons = cv2.line(image_reasons, (nhx1, nhy1), (nhx2, nhy2), 255, 2, cv2.LINE_AA )
                elif not right and K <0:
                    reasons += 1

                    image_reasons = cv2.line(image_reasons, (hx1, hy1), (hx2, hy2), 255, 2, cv2.LINE_4 )
                    image_reasons = cv2.line(image_reasons, (nhx1, nhy1), (nhx2, nhy2), 255, 2, cv2.LINE_4 )
    return reasons, image_reasons

