# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/huxiaoping/Desktop/demo/mainWindow.ui'
#
# Created: Mon Apr 24 19:15:53 2017
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1023, 812)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.videoFrame_1 = QtGui.QFrame(self.centralWidget)
        self.videoFrame_1.setGeometry(QtCore.QRect(20, 90, 981, 621))
        self.videoFrame_1.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.videoFrame_1.setFrameShape(QtGui.QFrame.StyledPanel)
        self.videoFrame_1.setFrameShadow(QtGui.QFrame.Raised)
        self.videoFrame_1.setLineWidth(10)
        self.videoFrame_1.setMidLineWidth(0)
        self.videoFrame_1.setObjectName(_fromUtf8("videoFrame_1"))
        self.DetectLabel = QtGui.QLabel(self.videoFrame_1)
        self.DetectLabel.setEnabled(True)
        self.DetectLabel.setGeometry(QtCore.QRect(0, 0, 981, 621))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.DetectLabel.setFont(font)
        self.DetectLabel.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.DetectLabel.setFrameShape(QtGui.QFrame.Box)
        self.DetectLabel.setFrameShadow(QtGui.QFrame.Plain)
        self.DetectLabel.setText(_fromUtf8(""))
        self.DetectLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.DetectLabel.setObjectName(_fromUtf8("DetectLabel"))
        self.label_notice = QtGui.QLabel(self.videoFrame_1)
        self.label_notice.setGeometry(QtCore.QRect(910, 30, 251, 101))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(159, 158, 158))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_notice.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_notice.setFont(font)
        self.label_notice.setText(_fromUtf8(""))
        self.label_notice.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_notice.setObjectName(_fromUtf8("label_notice"))
        self.label_1 = QtGui.QLabel(self.centralWidget)
        self.label_1.setGeometry(QtCore.QRect(210, 10, 691, 61))
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_1.setObjectName(_fromUtf8("label_1"))
        self.frame_1 = QtGui.QFrame(self.centralWidget)
        self.frame_1.setGeometry(QtCore.QRect(140, 730, 691, 61))
        self.frame_1.setFrameShape(QtGui.QFrame.Box)
        self.frame_1.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_1.setObjectName(_fromUtf8("frame_1"))
        self.startVideo = QtGui.QPushButton(self.frame_1)
        self.startVideo.setGeometry(QtCore.QRect(40, 10, 100, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.startVideo.setFont(font)
        self.startVideo.setObjectName(_fromUtf8("startVideo"))
        self.stopVideo = QtGui.QPushButton(self.frame_1)
        self.stopVideo.setGeometry(QtCore.QRect(210, 10, 100, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.stopVideo.setFont(font)
        self.stopVideo.setObjectName(_fromUtf8("stopVideo"))
        self.ResetVideo = QtGui.QPushButton(self.frame_1)
        self.ResetVideo.setGeometry(QtCore.QRect(390, 10, 100, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.ResetVideo.setFont(font)
        self.ResetVideo.setObjectName(_fromUtf8("ResetVideo"))
        self.NoDisplay = QtGui.QPushButton(self.frame_1)
        self.NoDisplay.setGeometry(QtCore.QRect(550, 10, 100, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.NoDisplay.setFont(font)
        self.NoDisplay.setObjectName(_fromUtf8("NoDisplay"))
        self.label_date = QtGui.QLabel(self.centralWidget)
        self.label_date.setGeometry(QtCore.QRect(800, 10, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_date.setFont(font)
        self.label_date.setObjectName(_fromUtf8("label_date"))
        self.label_week = QtGui.QLabel(self.centralWidget)
        self.label_week.setGeometry(QtCore.QRect(940, 10, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(17)
        self.label_week.setFont(font)
        self.label_week.setObjectName(_fromUtf8("label_week"))
        self.label_time = QtGui.QLabel(self.centralWidget)
        self.label_time.setGeometry(QtCore.QRect(850, 40, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_time.setFont(font)
        self.label_time.setObjectName(_fromUtf8("label_time"))
        self.frame_3 = QtGui.QFrame(self.centralWidget)
        self.frame_3.setGeometry(QtCore.QRect(20, 10, 280, 71))
        self.frame_3.setFrameShape(QtGui.QFrame.Box)
        self.frame_3.setFrameShadow(QtGui.QFrame.Plain)
        self.frame_3.setObjectName(_fromUtf8("frame_3"))
        self.label_2dcode = QtGui.QLabel(self.frame_3)
        self.label_2dcode.setGeometry(QtCore.QRect(10, 3, 61, 61))
        self.label_2dcode.setText(_fromUtf8(""))
        self.label_2dcode.setPixmap(QtGui.QPixmap(_fromUtf8("data/pku.png")))
        self.label_2dcode.setScaledContents(True)
        self.label_2dcode.setObjectName(_fromUtf8("label_2dcode"))
        self.label_website = QtGui.QLabel(self.frame_3)
        self.label_website.setGeometry(QtCore.QRect(94, 10, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.label_website.setFont(font)
        self.label_website.setObjectName(_fromUtf8("label_website"))
        self.label_email = QtGui.QLabel(self.frame_3)
        self.label_email.setGeometry(QtCore.QRect(90, 35, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.label_email.setFont(font)
        self.label_email.setObjectName(_fromUtf8("label_email"))
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "PDACS", None))
        self.label_1.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600;\">People Detection And Counting System</span></p></body></html>", None))
        self.startVideo.setText(_translate("MainWindow", "Start", None))
        self.stopVideo.setText(_translate("MainWindow", "Stop", None))
        self.ResetVideo.setText(_translate("MainWindow", "Reset", None))
        self.NoDisplay.setText(_translate("MainWindow", "Clear", None))
        self.label_date.setText(_translate("MainWindow", "date", None))
        self.label_week.setText(_translate("MainWindow", "week", None))
        self.label_time.setText(_translate("MainWindow", "time", None))
        self.label_website.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">尤安升      1300013023</span></p></body></html>", None))
        self.label_email.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; color:#128cac;\">youansheng@pku.edu.cn</span></p></body></html>", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

