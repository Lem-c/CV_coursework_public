import logging
import sys
from typing import List

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtGui import QIcon
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow, QFileDialog, QGraphicsScene
from PySide6.QtCore import QFile, QIODevice, Slot, Qt

import ui.ui_controller as ui_util
import util.subclass_helper as class_util
from stitches import ImageD
from stitches.image_blend import blendingImageList
from stitches.image_gain_compensations import setGainCompensation2Images, applyGain
from stitches.image_matching_list import MultiSIFTMatcher
from stitches.util import get_inner_max_rect_in_img, auto_image_slic
from util import *
from util.image_save import saveImageList, saveImageDList
from util.signal_broadcaster import WindowSignals


IS_DEBUG = False


class App(QMainWindow):

    def __init__(self, _ui):
        """
        Init the app object by a given ui path \n
        Generate QFile to create window \n
        Contains widget parameter to access child QObject \n
        ! Do not call any method through object !

        :param _ui: The relative path of .ui file
        """
        super(App, self).__init__()

        # read in ui file
        self.ui_path = _ui
        self.ui_file = QFile(self.ui_path)
        if not self.ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {self.ui_path}: {self.ui_file.errorString()}")
            sys.exit(-1)
        # load into window
        loader = QUiLoader()
        self.ui = loader.load(self.ui_file)
        # fix window size
        self.ui.setFixedSize(self.ui.width(), self.ui.height())
        if not self.ui:
            # print load fail message
            print(loader.errorString())
            sys.exit(-1)

        # Init all widgets in ui
        self.filePath = None
        self.imageList: List[ImageD] = None
        self.matcher: MultiSIFTMatcher = None
        # output result
        self.stitchResult: List[np.ndarray] = []
        # create canvas
        self.ui_scene = QGraphicsScene()
        # signal receiver
        self.signal = WindowSignals()

        self.__openApp()
        # close file IO stream
        self.ui_file.close()
        # show the ui
        self.ui.show()

    @Slot()
    def onImageStitchButtonClicked(self) -> None:
        """
        Main image stitch procedure callable method
        :return: None
        """
        logging.info("Image stitch clicked")
        self.__clean()
        if self.filePath is None or self.imageList is None:
            return

        l_ratio = self.__widgetSwitch("lRatio").value() * 0.01
        if l_ratio < 0.1:
            l_ratio = 0.1
        logging.info("Generating SIFT descriptor with L ratio: %f...", l_ratio)
        self.matcher = MultiSIFTMatcher(self.imageList, l_ratio)
        logging.info("Calculating match pair...")
        pair_list = self.matcher.getMatchPairs()
        # sort all matches in every pair
        pair_list.sort(key=lambda matched: len(matched.matcher), reverse=True)
        if IS_DEBUG:
            for pair in pair_list:
                class_util.matchesVisualization(pair)
        logging.info("Finding connect components...")
        # find connected components
        connect_components = class_util.getConnectComponents(pair_list)
        logging.info("Found %d connected components", len(connect_components))
        # build homographies for pairs
        class_util.generateHMatForImages(connect_components, pair_list)
        logging.info("Computing gain compensations...")
        setGainCompensation2Images(pair_list, connect_components)
        logging.info("Applying gain to images...")
        for image in connect_components[0]:
            image.image = applyGain(image)
        logging.info("Generating results...")
        # Add results
        for component in connect_components:
            self.stitchResult.append(blendingImageList(component))

        # slice stitched image
        for index, img in enumerate(self.stitchResult):
            if self.__widgetSwitch("isCrop").isChecked() is True:
                # reverse the value of slider | positive>0
                crop_degree = (10 - self.__widgetSwitch("cropDegree").value()) + 1
                logging.info("Applying auto cropping with level-"+str(crop_degree))
                self.stitchResult[index] = auto_image_slic(img, crop_degree)
            else:
                self.stitchResult[index] = get_inner_max_rect_in_img(img, _allow_x=0.8)

        logging.info("--------------------Image stitch successfully--------------- \n")
        # set result to be displayed on the graphic view object
        self.__setQWidget()
        # send a signal
        self.signal.isStitched_signal.emit("True")
        # find and activate the save button
        self.__widgetSwitch("save", True)

    @Slot()
    def onSaveButtonClicked(self):
        logging.info("saving clicked")
        # saveImageDList(self.imageList)
        saveImageList(self.stitchResult)
        logging.debug("Image saved successfully")

    # ------------------- Private methods -----------------------
    def __openApp(self):
        self.__initProcessButtons()
        self.__initMenuBar()
        self.__initIntSpinnerSelection()

    def __initProcessButtons(self):
        # Get the button from the window widget
        stitch_button = self.__widgetSwitch("stitch_image", False)
        save_button = self.__widgetSwitch("save", False)

        if stitch_button is not None and save_button is not None:
            # bind click listener
            stitch_button.clicked.connect(self.onImageStitchButtonClicked)
            save_button.clicked.connect(self.onSaveButtonClicked)
        else:
            raise Exception("Button objects not found!")

    def __initMenuBar(self):
        # Get the top menu and icon
        menu = self.ui.menuBar()
        openIcon = QIcon("./ui/icons/folder_128px.png")

        # Add menu option and its actions
        menu_file = menu.addMenu("File")

        file_action_open = menu_file.addAction("Open..")
        file_action_open.setIcon(openIcon)
        file_action_open.triggered.connect(self.__onMenuFileOpenClicked)

        menu_file.addSeparator()

        file_action_set = menu_file.addAction("Settings")

    def __initIntSpinnerSelection(self):
        # Get spin box
        spin_img_size = self.__widgetSwitch("spin_level")
        spin_interval = self.__widgetSwitch("interval")
        spin_sample_num = self.__widgetSwitch("imgNum")
        crop_degree = self.__widgetSwitch("cropDegree")

        spin_img_size.setToolTip("Set the sample image's size \nLess than 800 means no resize would be applied")
        spin_interval.setToolTip("Set the frame interval to pick a sample image \n"
                                 "[0, 144] | 0 means following original FPS")
        spin_sample_num.setToolTip("Number of total images used to stitch \n[4,21)")
        crop_degree.setToolTip("Cropping degree to the result image")

        spin_img_size.setRange(0, 4096)
        spin_interval.setRange(0, 144)
        spin_sample_num.setRange(4, 20)
        spin_img_size.setGroupSeparatorShown(True)  # Enable 1000 to be 1,000

    def __initLabels(self):
        # Get labels
        label_file_path = self.ui.findChild(QtWidgets.QLabel, "file_path")
        label_img_01 = self.ui.findChild(QtWidgets.QLabel, "image_1")
        label_img_02 = self.ui.findChild(QtWidgets.QLabel, "image_2")
        label_img_03 = self.ui.findChild(QtWidgets.QLabel, "image_3")
        label_img_04 = self.ui.findChild(QtWidgets.QLabel, "image_4")
        # add all labels into a list to check Null label
        image_labels = [label_file_path, label_img_01, label_img_02, label_img_03, label_img_04]
        if not ui_util.checkObjListContainsNull(image_labels):
            raise Exception("Can not find target label! Aborting..")

        # check null URL
        if self.filePath is not None and len(self.filePath) > 0:
            text = "Loaded: " + self.filePath[0]
            label_file_path.setText(text)

            # ui: func params | Get spin box
            spin_box = self.__widgetSwitch("spin_level")
            if spin_box.value() >= 800:
                image_size = spin_box.value()
            else:
                image_size = None

            frame_rate: int = self.__widgetSwitch("interval").value()

            img_num: int = self.__widgetSwitch("imgNum").value()

            logging.warning(str(frame_rate)+" fps; "+"Image resized into "+str(image_size))

            # read in video by given rate in a list
            vr = VideoReader(self.filePath[0], frame_rate, img_num)
            # get list saves all images with key points and feature info
            self.imageList = ui_util.getImageWithKeyPoints(vr.getImageList(), False, image_size)

            # set image for each label
            pixelMaps = ui_util.getImageWithKeyPoints(vr.getImageList())
            pixel_index = [0, 1, len(pixelMaps)-2, len(pixelMaps)-1]

            for label_index in range(4):
                image_labels[label_index+1].repaint()
                img_kp = pixelMaps[pixel_index[label_index]]
                img_kp = img_kp.scaled(image_labels[label_index+1].size(), aspectMode=Qt.KeepAspectRatio)
                image_labels[label_index+1].setPixmap(img_kp)

            logging.info("Video file loaded successfully with "+str(len(self.imageList))+" images.")
            if len(self.imageList) < img_num:
                logging.warning("The frame interval was set too large resulting to read enough number of images < "
                                + str(img_num))
            # enable the button
            self.__widgetSwitch("stitch_image", True)

    def __setQWidget(self):
        if self.stitchResult is None:
            return

        # resultDis | drawing
        img_widget = self.__widgetSwitch("resultDis")

        if img_widget is None:
            logging.error("Can not find QGraphicsView: check object name")
            return

        self.ui_scene.clear()
        # generate temp pixel mask
        pixMap = ui_util.getQPixelMapFromArray(self.stitchResult[0])
        self.ui_scene.addPixmap(pixMap)
        img_widget.setScene(self.ui_scene)
        img_widget.show()

    def __onMenuFileOpenClicked(self):
        logging.info("Opening file...")
        # select file
        self.filePath, _ = QFileDialog.getOpenFileNames(self.ui, 'Open Images', '../',
                                                        'Video files (*.mp4 *.avi)')
        if self.filePath is not None:
            self.__initParameters()
            self.__initLabels()

    def __widgetSwitch(self, _name: str, _flag: bool = True) -> object:
        """
        Obtain an QtWidget object that usually repeatedly obtained \n
        Avoid using this method frequently

        :param _name: The name of the widget
        :param _flag: Auto turn on/off
        :return: object: QtWidget
        """
        component = None

        if _name == "stitch_image":
            component = self.ui.findChild(QtWidgets.QPushButton, "stitch_image")
        elif _name == "save":
            component = self.ui.findChild(QtWidgets.QPushButton, "save")
        elif _name == "spin_level":
            component = self.ui.findChild(QtWidgets.QSpinBox, "spin_level")
        elif _name == "interval":
            component = self.ui.findChild(QtWidgets.QSpinBox, "interval")
        elif _name == "imgNum":
            component = self.ui.findChild(QtWidgets.QSpinBox, "imgNum")
        elif _name == "lRatio":
            component = self.ui.findChild(QtWidgets.QSlider, "lRatio")
        elif _name == "resultDis":
            component = self.ui.findChild(QtWidgets.QGraphicsView, "resultDis")
        elif _name == "isCrop":
            component = self.ui.findChild(QtWidgets.QCheckBox, "isCrop")
        elif _name == "cropDegree":
            component = self.ui.findChild(QtWidgets.QSlider, "cropDegree")

        if component is None:
            logging.error("Can not find button with name: '"+_name+"' !")
            exit(-1)

        component.setEnabled(_flag)

        return component

    def __initParameters(self):
        self.imageList = None
        self.__clean()

    def __clean(self):
        self.matcher = None
        self.stitchResult = []
