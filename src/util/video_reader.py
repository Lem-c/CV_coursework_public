import logging

import cv2


def checkIsVideoOpened(video):
    if not video.isOpened():
        raise Exception("'cv2.VideoCapture(_)' failed to open target video")

    return True


def readImgInList(_video, _frame_rate, _max_img_num):
    """
    Read given video clip into a list that saves fixed number of images

    :param _video: target video file path
    :param _max_img_num: saved number of image
    :param _frame_rate: save interval
    :return: list
    """
    frame_count = 1
    image_list = []

    if checkIsVideoOpened(_video):
        # TODO: check memory leak | brake condition
        while True:
            # read in image frame by frame
            ret, frame = _video.read()

            # Quit loop
            if ret is False or len(image_list) >= _max_img_num:
                break

            if frame_count == 2:
                image_list.append(frame)

            # save img into list by set interval
            if frame_count % _frame_rate == 0:
                image_list.append(frame)

            frame_count += 1

    return image_list


class VideoReader:

    def __init__(self, _video_path,
                 _load_frame: int = 0,
                 _max_img: int = 5):
        """
        Video read util object class \n
        Generate a list maintains a series of images \n
        Contains methods to obtain the image list and video detail

        :param _video_path: Video file path
        :param _load_frame: The frame rate of given video
        :param _max_img: How many images would be saved into the list
        """
        self.video = cv2.VideoCapture(_video_path)
        self.fps = _load_frame
        self.img_list = []

        checkIsVideoOpened(self.video)
        self.initParam(_max_img)
        self.video.release()

    def initParam(self, _img_list_size):
        if self.fps == 0:
            w, h, self.fps = self.getVideoInformation()
            # restrict the  resolution of the video
            if w < 800 or h < 600:
                logging.error("Video size too small!")
                raise Exception("The video does not contains enough information!")

        self.img_list = readImgInList(self.video, self.fps, _img_list_size)

    def getImageList(self):
        return self.img_list

    def getVideoInformation(self):
        w = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.video.get(cv2.CAP_PROP_FPS)
        logging.info("Video original fps: " + str(fps))
        return w, h, fps
