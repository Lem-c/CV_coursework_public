from PySide6.QtCore import Signal, QThread


class WindowSignals(QThread):
    isStitched_signal = Signal(str)

    def __init__(self, _SIFT_list=None):
        """
        Window signals class used to represent the signal transmitted \n
        Usage: local parameter of class:  Use emit/connect to send/obtain the signal
        """
        super().__init__()
        
        self.isChanged = False
        self.imgList = _SIFT_list

    def trigger(self):
        self.isChanged = True

    def reverse(self):
        self.isChanged = not self.isChanged

    def setSIFTImageList(self, _sift):
        self.imgList = _sift

    def run(self):
        if not self.isChanged:
            return
        # if triggered the button, start running
