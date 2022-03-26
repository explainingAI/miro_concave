# -*- coding: utf-8 -*-
""" Method for concave point detection from Miró et al. (2022)

Writen by: Miquel Miró Nicolau (UIB), 2020

"""
from concave import curvature as curv
from concave import regions, points

import numpy as np
import cv2


def weighted_median(data):
    """ Get the interest point inside the region.

    A regions has only one interest point. The interest point is the point is calculated with an
    approximation to a weighted median. The calculation is done with the cumulative sum of the
    curvature values. Our goal is the homogenous division in two classes of this cumulative sum.

    Args:
        data: Array with the curvature values.
    Returns:

    """
    cum_sum = np.cumsum(data)
    search = cum_sum[-1] / 2
    distances = np.abs(cum_sum - search)

    min_dist = np.argmin(distances)
    return min_dist


def concave_point_detector(contour, k: int, l_min: int, l_max: int, epsilon: float):
    """ Calculate the concave point.

    A concave point is a point with maximum negative curvature. In this case we calculate the
    concavity of the point with the original k-slope formulation. The k-slope it depens on the axis
    that is calculated and is as follows: k-slope_x = yi - y ( i + k) / xi - x ( i + k)

    And in the other axis is the inverse division. Once we get all the slopes of every point we
    need to calculate the curvature, that is the difference betwen the slope of the point n with
    the point n+k.

    Once we have the curvature we need to binarize the data. To do it we use a recursive method
    the method has an increasing threshold on each recusion. The stop condition is that the binary
    segment from the recursion is the lenght of the segment. After that to get the concave point
    we use the weighted median to get the interest points. Finally we check if are concave or
    convex.

    References:
        Under revision.

    Args:
        contour (contour): Object of the class contour containing all the information about the
            clump.
        k (int): Distance used to calculate the k-slope.
        l_min (int): Minimum longitude of the segment, used in the dynamic method.
        l_max (int): Maximum longitude of the segment.
        epsilon (int): Parameter for RDP approximation method.

    Returns:
        A list containing the index of every concave points in the contour.
    """
    interests_points = []

    contour_org = contour
    contour = np.copy(contour)

    # Simplification of the contour with RDP algorithm
    contour = cv2.approxPolyDP(contour, epsilon, True)
    contour = contour.reshape(-1, 1, 2)

    curvature = curv.k_curvature(contour, k)

    threshold = int(np.percentile(curvature, 25)) + 1
    curvature_binary = regions.threshold_data(curvature, threshold)

    # We check if there are at least one pixel of a region
    if curvature_binary.max() > 0:
        positions, length = regions.regions_of_interest(curvature_binary, l_min)
        positions, length = regions.refine_regions(positions, length, curvature, threshold, l_min,
                                                   l_max)

        for seg_pos, seg_length in zip(positions, length):
            interest_point = weighted_median(curvature[seg_pos: seg_pos + seg_length])
            interests_points.append(interest_point)

    concave = points.discriminate_interest_points(interests_points, k, contour).astype(int)
    concave = [points.get_nearest_point(contour_org, concave_point) for concave_point in concave]

    return concave
