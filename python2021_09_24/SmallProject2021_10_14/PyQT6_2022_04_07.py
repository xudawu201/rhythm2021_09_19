'''
Author: xudawu
Date: 2022-04-07 16:33:38
LastEditors: xudawu
LastEditTime: 2022-06-06 17:43:11
'''
from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(637, 364)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_01 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_01.setGeometry(QtCore.QRect(260, 150, 100, 24))
        self.pushButton_01.setObjectName("pushButton_01")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 80, 55, 16))
        self.label.setObjectName("label")
        self.pushButton_02 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_02.setGeometry(QtCore.QRect(410, 150, 100, 24))
        self.pushButton_02.setObjectName("pushButton_02")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 637, 22))
        self.menubar.setObjectName("menubar")
        self.menutype_here01 = QtWidgets.QMenu(self.menubar)
        self.menutype_here01.setObjectName("menutype_here01")
        self.menutype_here02 = QtWidgets.QMenu(self.menutype_here01)
        self.menutype_here02.setObjectName("menutype_here02")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actiontype_here_02 = QtGui.QAction(MainWindow)
        self.actiontype_here_02.setObjectName("actiontype_here_02")
        self.actiontype02 = QtGui.QAction(MainWindow)
        self.actiontype02.setObjectName("actiontype02")
        self.menutype_here02.addSeparator()
        self.menutype_here02.addAction(self.actiontype02)
        self.menutype_here01.addAction(self.menutype_here02.menuAction())
        self.menutype_here01.addAction(self.actiontype_here_02)
        self.menubar.addAction(self.menutype_here01.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_01.setText(_translate("MainWindow", "PushButton_01"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_02.setText(_translate("MainWindow", "PushButton_02"))
        self.menutype_here01.setTitle(_translate("MainWindow", "type here01"))
        self.menutype_here02.setTitle(_translate("MainWindow", "type here02"))
        self.actiontype_here_02.setText(_translate("MainWindow", "type here02"))
        self.actiontype02.setText(_translate("MainWindow", "type02"))
#QtWidgets????????????????????????UI????????????
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

def main():
    #QApplication??????PyQt5?????????GUI???????????????
    #sys.argv?????????????????????????????????????????????
    #QApplication???????????????????????????????????????list???tuple????????????,??????????????????????????????
    app = QApplication(sys.argv)

    MainWindow = QMainWindow()

    # app = QApplication(sys.argv)
    # ????????????????????????
    ui =Ui_MainWindow()
    # ?????????????????????setupUi
    ui.setupUi(MainWindow)


    # def click_success():
    #     print('click01 successful')
    # #??????????????????????????????,????????????????????????,???????????????
    # ui.pushButton_01.clicked.connect(click_success)

    #????????????,????????????
    ui.pushButton_01.clicked.connect(lambda: print("click01 successful"))

    #??????partial??????,???????????????
    from functools import partial

    # ui.pushButton.clicked.connect(partial(convert, ui))


    #????????????
    MainWindow.show()

    #??????exit()???????????????????????????????????????
    sys.exit(app.exec())

if __name__ == '__main__':
    main()