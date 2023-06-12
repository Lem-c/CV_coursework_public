from typing import List, Union, Optional

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from stitches import ImageD


def getQPixMapFromImageList(_list: List[np.ndarray]) -> List[QPixmap]:
    """
    Get a list saves ndarray as input \n
    Convert each element into QPixmap object and return a list saves them
    :param _list: Target image list
    :return: QPixmap list
    """
    pix_list = []

    for item in _list:
        q_image = getQImageFromArray(item)
        pix_map = QPixmap.fromImage(q_image)
        pix_list.append(pix_map)

    if len(pix_list) <= 1:
        raise Exception("Given image list contains too few image!")

    return pix_list


def getImageWithKeyPoints(_list: list, _flag=True, _size=None):
    """
    To get the pix map list that saves image with key points
    :param _size: The size of image
    :param _flag: The output type, optional
    :param _list: target list saves images: np.ndarray
    :return: list saves pixel maps
    """
    img_list: List[Union[np.ndarray, ImageD]] = []
    image_id = 0

    for image in _list:
        img_obj = ImageD(image, image_id, _re_size=_size)
        if _flag:
            img_list.append(img_obj.visualizationKeyPoints())
        else:
            img_list.append(img_obj)

        image_id += 1

    if not _flag:
        return img_list
    else:
        return getQPixMapFromImageList(img_list)


def checkObjListContainsNull(_obj_list: list) -> bool:
    """
    Check whether a given list, saves all objects, contains null obj
    :param _obj_list: list saves targets
    :return: bool
    """
    result = True

    for obj in _obj_list:
        result &= obj is not None

    return result


def getQImageFromArray(_img: np.ndarray) -> QImage:
    """
    Convert an array saves image in ndarray type to QImage object \n
    :param _img: target np.ndarray
    :return: QImage()
    """
    # Convert the numpy array to a QImage
    height, width, channels = _img.shape
    # convert bgr to rgb
    img_rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    bytesPerLine = channels * width
    qImg = QImage(img_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return qImg


def getQPixelMapFromQImage(_img: QImage) -> QPixmap:
    return  QPixmap.fromImage(_img)


def getQPixelMapFromArray(_img: np.ndarray) -> QPixmap:
    """
    Get a QPixel map object from a np.ndarray
    :param _img: The given image
    :return: QPixmap
    """
    return getQPixelMapFromQImage(getQImageFromArray(_img))


def signalSelector(_signal: str):
    if _signal == "True":
        return True
    else:
        return False
