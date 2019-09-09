# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'label_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_LabelDialog(object):
    def setupUi(self, LabelDialog):
        LabelDialog.setObjectName("LabelDialog")
        LabelDialog.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(LabelDialog)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = ImageDisplay(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.question = QtWidgets.QLineEdit(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.question.sizePolicy().hasHeightForWidth())
        self.question.setSizePolicy(sizePolicy)
        self.question.setObjectName("question")
        self.verticalLayout_2.addWidget(self.question)
        self.widget_2 = QtWidgets.QWidget(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.asparagus_name = QtWidgets.QLineEdit(self.widget_2)
        self.asparagus_name.setMinimumSize(QtCore.QSize(150, 0))
        self.asparagus_name.setMaximumSize(QtCore.QSize(150, 16777215))
        self.asparagus_name.setObjectName("asparagus_name")
        self.horizontalLayout.addWidget(self.asparagus_name)
        self.asparagus_no = QtWidgets.QSpinBox(self.widget_2)
        self.asparagus_no.setMinimumSize(QtCore.QSize(80, 0))
        self.asparagus_no.setMaximumSize(QtCore.QSize(100, 16777215))
        self.asparagus_no.setMaximum(999999999)
        self.asparagus_no.setObjectName("asparagus_no")
        self.horizontalLayout.addWidget(self.asparagus_no)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.previous_question = QtWidgets.QPushButton(self.widget_2)
        self.previous_question.setMaximumSize(QtCore.QSize(35, 16777215))
        self.previous_question.setObjectName("previous_question")
        self.horizontalLayout.addWidget(self.previous_question)
        self.next_question = QtWidgets.QPushButton(self.widget_2)
        self.next_question.setMaximumSize(QtCore.QSize(35, 16777215))
        self.next_question.setObjectName("next_question")
        self.horizontalLayout.addWidget(self.next_question)
        self.yes = QtWidgets.QPushButton(self.widget_2)
        self.yes.setObjectName("yes")
        self.horizontalLayout.addWidget(self.yes)
        self.no = QtWidgets.QPushButton(self.widget_2)
        self.no.setObjectName("no")
        self.horizontalLayout.addWidget(self.no)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.verticalLayout.addWidget(self.widget)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        LabelDialog.setCentralWidget(self.centralwidget)

        self.retranslateUi(LabelDialog)
        QtCore.QMetaObject.connectSlotsByName(LabelDialog)

    def retranslateUi(self, LabelDialog):
        _translate = QtCore.QCoreApplication.translate
        LabelDialog.setWindowTitle(_translate("LabelDialog", "Hand Label Assistant"))
        self.previous_question.setText(_translate("LabelDialog", "<<"))
        self.next_question.setText(_translate("LabelDialog", ">>"))
        self.yes.setText(_translate("LabelDialog", "Yes"))
        self.no.setText(_translate("LabelDialog", "No"))

from imagedisplay import ImageDisplay