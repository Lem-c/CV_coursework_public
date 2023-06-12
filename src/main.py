import logging
import sys

from PySide6.QtWidgets import QApplication

from ui.App import App

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG,
                        filename='IStitch_output.log',
                        filemode='a')

    app = QApplication(sys.argv)

    # init style | The path is start from where the call been called
    with open("./ui/style_sheet/main.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)

    self_app = App("./ui/sys_main.ui")

    app.exec()
