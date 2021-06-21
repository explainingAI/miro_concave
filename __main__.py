# -*- coding: utf-8 -*-
import cv2

from concave import method
from ellipse_fitting import cell


def main():
    path = "./in/ErythrocytesIDB2_01.png"

    # We extract the contours from the image
    img = cv2.imread(path)
    _, contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # We select one contour and we extract concave points
    concave_points = method.concave_point_detector(contours[0], k=9, l_min=6, l_max=25, epsilon=0.1)

    # From the concave point we fit ellipses to detect the cells
    cells, cells_type = cell.find_cells(contours, concave_points, img.shape)