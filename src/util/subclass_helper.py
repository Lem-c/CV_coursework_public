import logging
import warnings
from typing import List, Set

import numpy as np

from stitches import ImageD
from stitches.image_conn_pair import ImagePair


def getConnectComponents(_pair_list: List[ImagePair]) -> List[List[ImageD]]:
    """
    Find all connectable components of given pair list
    :param _pair_list:
    :return: List saves several image lists
    """

    if _pair_list is None or len(_pair_list) < 1:
        raise Exception("Null input pair list!")

    connected = []
    copy_pair_list = _pair_list.copy()
    connect_index = 0
    # using queue processing
    while len(copy_pair_list) > 0:
        # get first element on the top
        temp_pair = copy_pair_list.pop(0)
        # set connection tuple and get its size
        connected_component = {temp_pair.image_1, temp_pair.image_2}
        # TODO: The set initialize seems deprecated in this case, so do the 'size'
        size = len(connected_component)
        # check flag
        isStable = False

        # shaking loop
        while not isStable:
            counter = 0

            while counter < len(copy_pair_list):
                temp_pair = copy_pair_list[counter]
                if (checkIsSetContainsImage(temp_pair.image_1, temp_pair.image_2,
                                            connected_component)):
                    connected_component.add(temp_pair.image_1)
                    connected_component.add(temp_pair.image_2)
                    # inner pop without hold the return
                    copy_pair_list.pop(counter)
                else:
                    # not find connected
                    counter += 1

            # check stable state | wait to add to same numbers of elements
            isStable = len(connected_component) == size
            # update size
            size = len(connected_component)

        # After getting stable connect component, build list and add into result
        connected.append(list(connected_component))
        # update / assign group id for each image
        for img in connected_component:
            img.component_id = connect_index
        connect_index += 1

    return connected


def generateHMatForImages(_connect_components: List[List[ImageD]],
                          _pair_list: List[ImagePair],
                          _max_iter: int = 100,
                          _temp_H_mat: any = np.eye(3)) -> None:
    for item in _connect_components:
        # init matches from pair list
        item_matches = get_matches_pair(_pair_list, item)

        # generate set saves non-repetitive images
        img_set = set()
        iter_time = 0

        if len(item_matches) < 1:
            warnings.warn("Seems there is no matched components")

        # get first pair in the matches
        pair_match = item_matches[0]
        # re-calculate Homograph TODO: Test to remove this step
        pair_match.getHomographyResult()

        pair_size = len(_pair_list)

        sum_match_index_1 = sum(10 * (pair_size - i)
                                for i, match in enumerate(_pair_list)
                                if checkIsPairContainsImage(pair_match.image_1, match))
        sum_match_index_2 = sum(10 * (pair_size - i)
                                for i, match in enumerate(_pair_list)
                                if checkIsPairContainsImage(pair_match.image_2, match))

        # Assign homograph matrix for each image in a pair
        if sum_match_index_1 > sum_match_index_2:
            pair_match.image_1.transform_mat_H = np.eye(3)
            pair_match.image_2.transform_mat_H = pair_match.transform_mat
        else:
            pair_match.image_2.transform_mat_H = np.eye(3)
            # assign symmetric H matrix
            pair_match.image_1.transform_mat_H = np.linalg.inv(pair_match.transform_mat)

        # add updated images into set | No deprecated insert
        img_set.add(pair_match.image_1)
        img_set.add(pair_match.image_2)

        while len(img_set) < len(item):
            iter_time += 1
            if iter_time >= _max_iter:
                logging.warning("Early quit: (Increasing iter times to avoid this)")
                break
            # update images' transform matrix
            for pair_match in item_matches:
                # get H mat for un-processed image in pairs and add into set
                if pair_match.image_1 in img_set and pair_match.image_2 not in img_set:
                    pair_match.getHomographyResult()
                    homography = pair_match.transform_mat @ _temp_H_mat
                    pair_match.image_2.transform_mat_H = \
                        pair_match.image_1.transform_mat_H @ homography
                    img_set.add(pair_match.image_2)
                    break

                elif pair_match.image_1 not in img_set and pair_match.image_2 in img_set:
                    pair_match.getHomographyResult()
                    homography = np.linalg.inv(pair_match.transform_mat) @ _temp_H_mat
                    pair_match.image_1.transform_mat_H = \
                        pair_match.image_2.transform_mat_H @ homography
                    img_set.add(pair_match.image_1)
                    break

    logging.log(level=logging.INFO, msg="Homographies build finished", exc_info=True)


def checkIsListContainsImage(_a: ImageD,
                             _b: ImageD,
                             _pair: List[ImageD]) -> bool:
    """
    Check whether a list contains any other one image provided
    :param _a: image 01
    :param _b: image 02
    :param _pair: List of two images
    :return: bool
    """
    if _a in _pair or _b in _pair:
        return True

    return False


def checkIsPairContainsImage(_img: ImageD, _pair: ImagePair) -> bool:
    return _pair.image_1 == _img or _pair.image_2 == _img


def checkIsSetContainsImage(_a: ImageD,
                            _b: ImageD,
                            _pair: Set[ImageD]) -> bool:
    """
    Check whether a list contains any other one image provided
    :param _a: image 01
    :param _b: image 02
    :param _pair: List of two images
    :return: bool
    """
    if _a in _pair or _b in _pair:
        return True

    return False


def get_matches_pair(_pair_list: List[ImagePair], _img_list: List[ImageD]) -> List[ImagePair]:
    """
    Get list saves matches pair by checking whether contains the src image in pair
    :param _pair_list: Target pair list
    :param _img_list: The list saves images
    :return: item_matches
    """
    item_matches = []
    for pair in _pair_list:
        if pair.image_1 in _img_list:
            item_matches.append(pair)

    return item_matches


def matchesVisualization(_pair: ImagePair) -> None:
    """
    Debug used method that draws matches on the image \n
    TODO: The visual experience seems not as expected
    :param _pair:
    :return:
    """
    import cv2
    from util.image_save import imageDisplay

    matchesMask = _pair.mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inlines
                       flags=2)

    m_img = cv2.drawMatches(_pair.image_1.image,
                            _pair.image_1.key_points,
                            _pair.image_2.image,
                            _pair.image_2.key_points,
                            _pair.matcher, None, **draw_params)

    imageDisplay(m_img, _shrink=0.25)


def generateMidFirstRandomIndex(_size: int) -> List[int]:
    """
    Generate a list contains random shuffled index with given size \n
    However, the original mid-index would always be placed at the first one
    :param _size: size of the output
    :return: list
    """
    import random
    new_list = [i-1 for i in range(1, _size + 1)]
    mid = new_list[int(len(new_list)/2) + 1]

    random.shuffle(new_list)
    mid_index = new_list.index(mid)
    # swap
    if mid_index != 0:
        new_list[0], new_list[mid_index] = new_list[mid_index], new_list[0]

    return new_list


def get_sub_classes(class_):
    """Display subclass using recursion"""
    for subclass in class_.__subclasses__():
        print(subclass)
        if len(class_.__subclasses__()) > 0:
            get_sub_classes(subclass)
