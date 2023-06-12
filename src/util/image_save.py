import os
import datetime
from typing import List

import cv2
import numpy as np

from stitches import ImageD

base_path = "./"


def saveImageList(_images: List[np.ndarray]) -> None:
    os.makedirs(os.path.join(base_path, "results"), exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i, result in enumerate(_images):
        cv2.imwrite(os.path.join(base_path, "results", f"panorama_{current_time}_{i}.jpg"), result)


def saveImageDList(_images: List[ImageD]) -> None:
    img_list = []
    for item in _images:
        img_list.append(item.image)

    saveImageList(img_list)


def convertImageList2Array(_images: List[ImageD]) -> List[np.ndarray]:
    result = []
    for img in _images:
        result.append(img.image)

    return result


def imagesDisplay(_images: List[np.ndarray]):
    for img in _images:
        cv2.imshow('Images', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def imageDisplay(_img, _shrink: float = 0.5):
    img = _img
    cv2.resize(img, (0, 0), fx=_shrink, fy=_shrink, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Images', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
