import logging

import cv2
import numpy as np


def imageResize(_img: np.ndarray, _size):
    """
    Method used to resize a target image into given size \n
    The parameter size is a single int, in some terms, this method can enlarge the image
    :param _img: The target image
    :param _size: The target size: decided by the original image
    :return: The scaled image
    """
    image = _img

    if _size is None:
        return image

    h, w = _img.shape[:2]
    if max(w, h) > _size:
        if w > h:
            image = cv2.resize(_img, (_size, int(h * _size / w)))
        else:
            image = cv2.resize(_img, (int(w * _size / h), _size))

    return image


def getFeatures(_img: np.ndarray, _mask=None):
    interpreter = cv2.SIFT_create()
    return interpreter.detectAndCompute(_img, _mask)


class ImageD:

    def __init__(self, _img: np.ndarray, _id: int, _re_size: int = None):
        """
        The image object class that used to save key data for Panorama generation \n
        Contains parameters for image stitching \n

        :param _img: np.ndarray
        :param _re_size: int, optional
        """
        if _img is None:
            raise Exception("Image object can not construct from a null imagee!")

        self.image = _img
        self.key_points = None                                  # calculated homograph matrix
        self.features = None                                    # features of image
        self.match_id: int = _id                                # ref in the match's id
        self.component_id: int = 0                              # index in connect component list
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)    # obtained gain compensation
        # (3*3) to fit the requirement of cv2.warpPerspective 's transform Mat
        self.transform_mat_H: np.ndarray = np.eye(3)            # The homograph result matrix

        # base parameter init
        self.__initialize(_re_size)

    def __initialize(self, _size):
        """
        If the '_re_size' parameter is set in constructor \n
        Resize the sample image to target size
        :param _size: target width : scale in ratio
        :return: None
        """
        self.image = imageResize(self.image, _size)
        self.key_points, self.features = getFeatures(self.image, None)

    def visualizationKeyPoints(self, isShow=False):
        if self.key_points is None:
            return

        image_with_key_points = cv2.drawKeypoints(self.image, self.key_points, None)

        if isShow:
            # Display the image with key points
            cv2.imshow('Image Key points', image_with_key_points)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image_with_key_points
