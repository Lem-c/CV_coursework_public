"""
@ref: https://github.com/CorentinBrtx/image-stitching
@ref: https://blog.csdn.net/qq_21420941/article/details/109249528
Partial used for matrix calculation
"""
import logging
from typing import List, Tuple

import numpy as np


def apply_homography(H: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply a homography to a point.

    Parameters
    ----------
    H : np.ndarray,
        Homography matrix.
    point : np.ndarray,
        Point to apply the homography to, with shape (2,1).

    Returns
    -------
    new_point : np.ndarray,
        Point after applying the homography, with shape (2,1).
    """
    point = np.asarray([[point[0][0], point[1][0], 1]]).T
    new_point = H @ point
    return new_point[0:2] / new_point[2]


def apply_homography_list(H: np.ndarray, points: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply a homography to a list of points.

    Parameters
    ----------
    H : np.ndarray
        Homography matrix.
    points : List         of points to apply the homography to, each with shape (2,1).

    Returns
    -------
    new_points : List
        List of points after applying the homography, each with shape (2,1).
    """
    return [apply_homography(H, point) for point in points]


def get_new_corners(image: np.ndarray, H: np.ndarray) -> List[np.ndarray]:
    """
    Get the new corners of an image after applying a homography.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    H : np.ndarray
        Homography matrix.

    Returns
    -------
    corners : List[np.ndarray]
        Corners of the image after applying the homography.
    """
    # get top and bottom four vertexes
    top_left = np.asarray([[0, 0]]).T
    top_right = np.asarray([[image.shape[1], 0]]).T
    bottom_left = np.asarray([[0, image.shape[0]]]).T
    bottom_right = np.asarray([[image.shape[1], image.shape[0]]]).T

    return apply_homography_list(H, [top_left, top_right, bottom_left, bottom_right])


def get_offset(corners: List[np.ndarray]) -> np.ndarray:
    """
    Get offset matrix so that all corners are in positive coordinates.

    Parameters
    ----------
    corners : List[np.ndarray]
        List of corners of the image.

    Returns
    -------
    offset : np.ndarray
        Offset matrix.
    """
    top_left, top_right, bottom_left = corners[:3]
    return np.array(
        [
            [1, 0, max(0, -float(min(top_left[0], bottom_left[0])))],
            [0, 1, max(0, -float(min(top_left[1], top_right[1])))],
            [0, 0, 1],
        ],
        np.float32,
    )


def get_new_size(corners_images: List[List[np.ndarray]]) -> Tuple[int, int]:
    """
    Given the four vertexes of an image \n
    Calculate the max width and possible height of the stitched image \n
    Fit the first smallest value to get (w, h) from x and y

    :param corners_images: List of corners of the images (i.e. corners_images[i] is the list of corners of image i).
    :return: (width, height), Size of the image
    """

    top_right_x = np.max([corners_image[1][0] for corners_image in corners_images])
    bottom_right_x = np.max([corners_images[3][0] for corners_images in corners_images])

    bottom_left_y = np.max([corners_images[2][1] for corners_images in corners_images])
    bottom_right_y = np.max([corners_images[3][1] for corners_images in corners_images])

    width = int(np.ceil(max(bottom_right_x, top_right_x)))
    height = int(np.ceil(max(bottom_right_y, bottom_left_y)))

    width = min(width, 5120)
    height = min(height, 4000)

    return width, height


def get_new_parameters(
    panorama: np.ndarray, image: np.ndarray, H: np.ndarray
) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Get the new size of the image and the offset matrix.

    Parameters
    ----------
    panorama : np.ndarray
        Current panorama.
    image : np.ndarray
        Image to add to the panorama.
    H : np.ndarray
        Homography matrix for the image.

    Returns
    -------
    size, offset :  Tuple[Tuple[int, int], np.ndarray]
        Size of the new image and offset matrix.
    """
    corners = get_new_corners(image, H)
    added_offset = get_offset(corners)
    # add offset to make coordinates positive
    corners_image = get_new_corners(image, added_offset @ H)
    if panorama is None:
        size = get_new_size([corners_image])
    else:
        corners_panorama = get_new_corners(panorama, added_offset)
        size = get_new_size([corners_image, corners_panorama])

    return size, added_offset


def single_weights_array(size: int) -> np.ndarray:
    """
    Create a linear distribution 1D weights array. \n
    target array would be like: [0, 0.25, 0.5, 1, 0.5, 0.25, 0] \n
    :param size: Size of the result array.
    :return: 1D weights array.
    """

    if size % 2 == 1:
        return np.concatenate(
            [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
        )
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])


def single_weights_matrix(shape: Tuple[int]) -> np.ndarray:
    """
    Create a 2D weights' matrix.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the matrix.

    Returns
    -------
    weights : np.ndarray
        2D weights' matrix.
    """
    return (
        single_weights_array(shape[0])[:, np.newaxis]
        @ single_weights_array(shape[1])[:, np.newaxis].T
    )


def get_inner_max_rect_in_img(_image: np.ndarray,
                              _is_gray: bool = False,
                              _allow_x: float = 0.9) -> np.ndarray:
    """
    Method used to find the bottom inner connected largest rectangle \n
    Partial solution of image cropping
    :param _image: Target image
    :param _is_gray: whether the target is already in gray channel
    :param _allow_x: the minimum width of output comparing with original image
    :return: Numpy.ndarray
    """
    import cv2

    img = _image
    if not _is_gray:
        # convert input into gray channel
        img = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)

    # binary threshold process
    ret, image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    np.set_printoptions(threshold=np.inf)
    logging.info(str(image))

    # obtain contours and properties(ignored)
    # mode: RETR_CCOMP to save a two-level hierarchy => external boundaries and internal holes
    vertexes, _ = cv2.findContours(image,
                                   cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contour = vertexes[0].reshape(len(vertexes[0]), 2)
    # result rectangle
    rects = set()
    # generate all possible rectangles
    for i, coord in enumerate(contour):
        x_1, y_1 = coord
        for j in range(len(contour)):
            x_2, y_2 = contour[j]
            area = abs(y_2-y_1) * abs(x_2-x_1)
            rects.add(((x_1, y_1), (x_2, y_2), area))

    rect_sort = sorted(rects, key=lambda x: x[2], reverse=True)

    if not rect_sort:
        logging.error("Exception in finding contours...")
        exit(-1)

    is_largest_rect = False
    index = 0
    left_top = None
    right_bottom = None

    # find valid rect with the largest area
    while index < len(rect_sort) and not is_largest_rect:
        rect = rect_sort[index]
        left_top = rect[0]
        right_bottom = rect[1]

        is_valid_rect = False
        # find valid x
        min_x = min(left_top[0], right_bottom[0])
        while min_x < max(left_top[0], right_bottom[0]) + 1 and not is_valid_rect:
            if any(_image[left_top[1], min_x]) != 0 and any(_image[right_bottom[1], min_x]) != 0:
                is_valid_rect = True
            min_x += 1

        # find valid y
        min_y = min(left_top[1], right_bottom[1])
        while min_y < max(left_top[1], right_bottom[1]) + 1 and not is_valid_rect:
            if any(_image[min_y, left_top[0]]) != 0 and any(_image[min_y, right_bottom[0]]) != 0:
                is_valid_rect = True
            min_y += 1

        if is_valid_rect:
            logging.warning("Find valid inner rectangle: _index: "+str(index))
            is_largest_rect = True

        index += 1

    logging.info("Getting sliced rectangle from the image...\n")
    catercorner = [left_top, right_bottom]
    xs = [p[1] for p in catercorner]
    ys = [p[0] for p in catercorner]
    img_crop = _image[min(xs):max(xs), min(ys):max(ys)]

    # if the stitched image is too small, recall the slice operation
    if max(xs) < int(_allow_x * _image.shape[0]):
        img_crop = _image

    return img_crop


def auto_image_slic(_image: np.ndarray,
                    _tolerance: int = 5):
    import cv2
    from util.image_save import imageDisplay

    ret, image_array = cv2.threshold(_image, 0, 255, cv2.THRESH_BINARY)

    row_sums = np.sum(image_array == 255, axis=1)
    col_sums = np.sum(image_array == 255, axis=0)

    top_rows = np.where(row_sums >= np.max(row_sums) - 255 * _tolerance)[0]
    top_cols = np.where(col_sums >= np.max(col_sums) - 255 * _tolerance)[0]

    min_row = np.min(top_rows)
    max_row = np.max(top_rows)
    min_col = np.min(top_cols)
    max_col = np.max(top_cols)

    return _image[min_row:max_row + 1, min_col:max_col + 1]
