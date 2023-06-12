from typing import List, Tuple

import cv2
import numpy as np

from stitches import ImageD
from stitches.util import get_new_parameters, single_weights_matrix
from util.subclass_helper import generateMidFirstRandomIndex


def blendingImageList(_images: List[ImageD],
                      _old_panorama: np.ndarray = None,
                      _weight: np.ndarray = None,
                      _offset: np.ndarray = np.eye(3),
                      _is_multi_band: bool = False) -> np.ndarray:
    """
    Linear blending by iterate each image and stitch them together \n
    Generate relative weight mask to perform transformation according to offset*H \n
    TODO: can not handling seams blur

    :param _is_multi_band: whether current blending is multi band blending
    :param _images: list saves orderly images in waiting list
    :param _old_panorama: If there is a given panorama(Usually None), optional
    :param _weight: Initialization pixel weight for image
    :param _offset: padding
    :return:
    """
    # init params
    panorama = _old_panorama
    weights = _weight
    offset = _offset

    if _is_multi_band:
        panorama = cvMultiBandBlending(_images, panorama)
    else:
        # for image in _images:
        #     # update parameters in time
        #     panorama, offset, weights = generateNewPanoramas(panorama, image, offset, weights)
        index = generateMidFirstRandomIndex(len(_images))
        for i in range(len(index)):
            print(index[i])
            panorama, offset, weights = generateNewPanoramas(panorama, _images[index[i]], offset, weights)

    return panorama


def generateNewPanoramas(_panorama: np.ndarray,
                         _image: ImageD,
                         _offset: np.ndarray,
                         _weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_homography_mat = _offset @ _image.transform_mat_H
    expend_size, new_offset = get_new_parameters(_panorama,
                                                 _image.image,
                                                 new_homography_mat)

    expend_image = cv2.warpPerspective(_image.image,
                                       new_offset @ new_homography_mat,
                                       expend_size)

    # check if there is an existed panorama
    if _panorama is not None:
        panorama = cv2.warpPerspective(_panorama, new_offset, expend_size)
        weights = cv2.warpPerspective(_weights, new_offset, expend_size)
    else:
        # generate new result
        panorama = np.zeros_like(expend_image)
        weights = np.zeros_like(expend_image)
    # get normalization distribution map of image
    weights_img = single_weights_matrix(_image.image.shape)
    # insert weight as new axis into the image | apply transform either
    weights_img_1d = cv2.warpPerspective(weights_img, new_offset @ new_homography_mat, expend_size)[:, :, np.newaxis]
    # repeat the weight in the third-axis to build 3d weight map
    weights_img = np.repeat(weights_img_1d, 3, axis=2)

    # Normalize the weights by  applying one another plus division when adding result is not zero
    weights_norm = np.divide(weights, (weights + weights_img),
                             where=(weights + weights_img) != 0)

    panorama = np.where(
        np.logical_and(
            np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
            np.repeat(np.sum(expend_image, axis=2)[:, :, np.newaxis], 3, axis=2) == 0
        ),
        0,
        expend_image * (1 - weights_norm) + panorama * weights_norm,
    ).astype(np.uint8)
    offset = new_offset @ _offset
    # added current weight to previous one
    weights = (weights + weights_img) / (weights + weights_img).max()

    return panorama, offset, weights


def cvMultiBandBlending(_images: List[ImageD],
                        _old_panorama: np.ndarray = None,
                        _blender=None) -> np.ndarray:
    # generate initial blender
    blender = cv2.detail.Blender_createDefault(cv2.detail.BLENDER_MULTI_BAND,
                                               try_gpu=False)
    panorama = _old_panorama
    offset = np.eye(3)
    weights = None

    return panorama

