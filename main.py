# -*- coding: utf-8 -*-

from mainWindow import *
from PyQt4.QtGui import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
