import logging
import cv2
from typing import List

from stitches import ImageD
from stitches.image_conn_pair import ImagePair


class MultiSIFTMatcher:
    def __init__(self, _images: List[ImageD], _l_ratio=0.7):
        """
        Find the matched features of multi images built by 'sift' feature method \n
        Using knn to find possible images with the nearest distance

        :param _images: list saves features calculated images
        :param _l_ratio: ratio used for the Lowe's ratio test, default 0.7
        """
        self.image_list = _images
        self.matches = {image.match_id: {} for image in _images}
        self.ratio = _l_ratio

    def computeMatches(self, _imgA: ImageD, _imgB: ImageD,
                       _k=2):
        """
        Using BF distance comparing method to get matcher \n
        Save ref index of two points with the closest distance

        :param _imgA: A feature obtained ImageD
        :param _imgB: Another feature obtained ImageD
        :param _k: num of knn neighbors
        :return: list saves matched ref index
        """

        # build matcher and sift match list
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        knnMatched = matcher.knnMatch(_imgA.features, _imgB.features, _k)

        # result save
        matches = []

        for rawIndex in knnMatched:
            # Remove false matching
            # 'rawIndex' contains two neighbors: first close and second close
            # When the distance ratio smaller than the set ratio
            if len(rawIndex) == 2 and rawIndex[0].distance < self.ratio * rawIndex[1].distance:
                # save the reference of two images in their feature space
                # reverse error: matches.append((rawIndex[0].trainIdx, rawIndex[0].queryIdx))
                matches.append(rawIndex[0])

        return matches

    def assignMatches(self, _imgA: ImageD, _imgB: ImageD):
        if _imgB.match_id not in self.matches[_imgA.match_id]:
            new_matches = self.computeMatches(_imgA, _imgB)
            self.matches[_imgA.match_id][_imgB.match_id] = new_matches
        # TODO: Can return local variable
        return self.matches[_imgA.match_id][_imgB.match_id]

    def getMatchPairs(self, _pair_size=6):
        """
        Sorting the length of image's matches list in descending order \n
        Obtaining a list saves possible paris of each paired two images \n
        After checking whether is validated? Add into list : Nothing

        :param _pair_size: The num of matched feature in the top of list
        :return: A list saves {pairs: object}
        """
        result = []

        for index, image in enumerate(self.image_list):
            # get sorted matches according two adjacent pictures
            comp_images = self.image_list[:index] + self.image_list[index+1:]
            # Pick top '_pair_size=6' matches to process
            comp_list = sorted(comp_images,
                               key=lambda img_1, img_2=image: len(self.assignMatches(img_2, img_1)),
                               reverse=True)[:_pair_size]
            # get pair groups by homogeneous matrix
            for img_matched_features in comp_list:
                if self.image_list.index(img_matched_features) > index:
                    # TODO: The match function has error leads to following 'matcher' error
                    temp_pair = ImagePair(image, img_matched_features,
                                          _isInit=False,
                                          _matches=self.assignMatches(image, img_matched_features))
                    if temp_pair.canConcatenateCurrentPair():
                        logging.debug("Matched image: (" + str(image.match_id) + ", "
                                      + str(img_matched_features.match_id) + ")")
                        result.append(temp_pair)

        return result

