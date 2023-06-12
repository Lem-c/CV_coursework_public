import warnings
from typing import List, Tuple

import numpy as np

import util.subclass_helper as class_util

from stitches import ImageD
from stitches.image_conn_pair import ImagePair


def setGainCompensation2Images(_pair_list: List[ImagePair],
                               _connect_components: List[List[ImageD]]) -> None:
    for _ in _connect_components:
        component_matches = class_util.get_matches_pair(_pair_list, _connect_components[0])
        # TODO: return value of following method would be ignored
        assignGain2List(_connect_components[0], component_matches)


def assignGain2List(_images: List[ImageD], _pair_matches: List[ImagePair],
                    _sigma_n: float = 10.0,
                    _sigma_g: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:

    # init value saving list
    coefficient = []
    results = []

    for i, img in enumerate(_images):
        # generate a list saves n-th [0,0,0] sub-array, n is the size of '_images'
        temp_coefficient = [np.zeros(3) for _ in range(len(_images))]
        # init result
        result = np.zeros(3)

        for pair in _pair_matches:
            # check same image in current pair
            if img == pair.image_1:
                params = previous_intense(pair, _sigma_n, _sigma_g)
                temp_coefficient[i] += params[0]

                pos = _images.index(pair.image_2)
                temp_coefficient[pos] -= params[1]

                result += params[2]

            elif img == pair.image_2:
                params = previous_intense(pair, _sigma_n, _sigma_g, False)
                temp_coefficient[i] += params[0]

                pos = _images.index(pair.image_1)
                temp_coefficient[pos] -= params[1]

                result += params[2]

        # Update list
        coefficient.append(temp_coefficient)
        results.append(result)

    # Convert to ndarray
    coefficient = np.array(coefficient)
    results = np.array(results)
    # generate gain array
    gain = np.ones_like(results)

    for dim in range(coefficient.shape[2]):
        temp_coefficient = coefficient[:, :, dim]
        temp_result = results[:, dim]

        gain[:, dim] = np.linalg.solve(temp_coefficient, temp_result)

    max_pixel_intensity = np.max([image.image for image in _images])

    # Normalization
    if gain.max() * max_pixel_intensity > 255:
        # convert to 255%
        gain = gain / (gain.max() * max_pixel_intensity) * 255
    # assign gain to image
    for i, img in enumerate(_images):
        img.gain = gain[i]

    return coefficient, results


def previous_intense(pair_match: ImagePair,
                     _sig_n, _sig_g,
                     flag=True) -> tuple:

    if flag:
        pos_i = pair_match.overlap.sum() * \
                ((2 * pair_match.Iab ** 2 / _sig_n ** 2) + (1 / _sig_g ** 2))
    else:
        pos_i = pair_match.overlap.sum() * \
                ((2 * pair_match.Iba ** 2 / _sig_n ** 2) + (1 / _sig_g ** 2))

    pair_i = ((2 / _sig_n ** 2) *
              pair_match.overlap.sum() * pair_match.Iab * pair_match.Iba)

    added = pair_match.overlap.sum() / _sig_g ** 2

    return pos_i, pair_i, added


def applyGain(_image: ImageD):
    """
    Apply gain to current image
    :param _image: The target image
    :return: new image with gain compensation
    """
    if _image.gain is None:
        warnings.warn("None calculated gain would lead to NULL image")

    return (_image.image * _image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)
