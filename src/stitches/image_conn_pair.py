import logging
import warnings
from typing import Optional, List

import cv2
import numpy as np

from stitches import ImageD

'''
    Static parameters defined here
'''
RANSAC_MAX_ITER: int = 1000


def getComparisonAlpha(_bias: float) -> (float, float):
    """
    The values in equation: alpha + beta*_
    :param _bias: base change degree
    :return: tuple saves two values
    """
    base_alpha = 8
    base_beta = 0.2

    return base_alpha+_bias, base_beta*_bias


class ImagePair:

    def __init__(self, _img1: ImageD, _img2: ImageD,
                 _isInit: bool = False,
                 _matches: Optional[List] = None):
        self.image_1 = _img1
        self.image_2 = _img2

        # The overlapped area of image_1 and image_2
        self.overlap = None
        self.overlap_area = None

        self.img_1_key_points = None
        self.img_2_key_points = None

        self.matcher = _matches
        self.transform_mat = None
        self.mask = None

        # instance variables
        self._I_pre_last = None
        self._I_last_pre = None

        if _isInit:
            self.getHomographyResult()

    @property
    def Iab(self):
        if self._I_pre_last is None:
            self.initInstances()
        return self._I_pre_last

    @property
    def Iba(self):
        if self._I_last_pre is None:
            self.initInstances()
        return self._I_last_pre

    def getHomographyResult(self, error_threshold: float = 5) -> None:
        """
        The method of calculating homography matrix \n
        Can be call in the constructor

        Using lib: cv2.findHomography(
            key_point_1,
            key_point_2,
            #CV_RANSAC: A robust approach based on RANSAC,
            error_threshold,
            max_iter: this is ignored in this case
        )

        :param error_threshold: The error between the point of the original image and
                                the corresponding point on the target image after transformation
        """

        if self.matcher is None:
            raise Exception("Matches is Null!")

        if self.img_1_key_points is not None or self.img_2_key_points is not None:
            logging.warning("Existed key point data would be over-wrote")

        # obtain both match point in float array
        self.img_1_key_points = np.float32(
            [self.image_1.key_points[match.queryIdx].pt for match in self.matcher]
        )
        self.img_2_key_points = np.float32(
            [self.image_2.key_points[match.trainIdx].pt for match in self.matcher]
        )

        # assign value of transform matrix and mask using train kp
        self.transform_mat, self.mask = cv2.findHomography(
            self.img_2_key_points,
            self.img_1_key_points,
            cv2.RANSAC,
            error_threshold,
            maxIters=RANSAC_MAX_ITER
        )

    def calculateOverlap(self) -> None:
        """
        Calculate the overlapped region of two images \n

        :return: None
        """
        # If params are not initialized
        if self.transform_mat is None:
            self.getHomographyResult()

        # get same shape 1d array filled with '1'
        # TODO: data type should be defined here as np.uint8
        one_1 = np.ones_like(self.image_1.image[:, :, 0], dtype=np.uint8)
        one_2 = np.ones_like(self.image_2.image[:, :, 0], dtype=np.uint8)

        # calculate the perspective transformed mask using reversed size(w, h) of 'image_1'
        perspective_trans = cv2.warpPerspective(one_2,
                                                self.transform_mat,
                                                one_1.shape[::-1])

        # Assign the overlap area
        self.overlap = one_1 * perspective_trans
        self.overlap_area = self.overlap.sum()

    def canConcatenateCurrentPair(self, _bias: int = 1) -> bool:
        """
        Check whether the two images in the pair are valid \n
        Calculate the
        :return: whether current pair is valid: bool
        """
        if self.overlap is None:
            self.calculateOverlap()

        # double check
        if self.mask is None:
            self.getHomographyResult()

        # target in the overlapped area, finding matches(first column, second column)
        matches_in_overlap = self.img_1_key_points[
            self.overlap[
                self.img_1_key_points[:, 1].astype(np.int64),
                self.img_1_key_points[:, 0].astype(np.int64),
            ]
            == 1
        ]
        # TODO: Parameter swift can be made here
        a, b = getComparisonAlpha(_bias)
        # Maximum allowable re-projection error threshold
        size = a + b*matches_in_overlap.shape[0]

        return self.mask.sum() > size

    def initInstances(self) -> None:
        """
        Calculate the intensity of two images in their overlap area \n
        Should be used in gain compensation step \n
        TODO: No method handles un-matched images (No overlap)
        :return: None
        """

        if self.overlap is None:
            warnings.warn("Overlap has not been set before: check allocation")
            self.calculateOverlap()

        # get reverse perspective transformed mask by [1::-1]
        inv_perspective_trans = cv2.warpPerspective(
            self.overlap, np.linalg.inv(self.transform_mat), self.image_2.image.shape[1::-1]
        )

        if self.overlap.sum() == 0:
            warnings.warn("No overlap calculated in current pair")

        # Convert 2d array to 3-channels gray scale array
        self._I_pre_last = (np.sum(
            self.image_1.image * np.repeat(self.overlap[:, :, np.newaxis], 3, axis=2),
            axis=(0, 1)) / self.overlap.sum())
        self._I_last_pre = (np.sum(
            self.image_2.image * np.repeat(inv_perspective_trans[:, :, np.newaxis], 3, axis=2),
            axis=(0, 1)) / inv_perspective_trans.sum())
