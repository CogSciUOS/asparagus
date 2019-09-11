# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'label_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LabelDialog(object):
    def setupUi(self, LabelDialog):
        LabelDialog.setObjectName("LabelDialog")
        LabelDialog.resize(1425, 980)
        self.centralwidget = QtWidgets.QWidget(LabelDialog)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_3.addWidget(self.line)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = ImageDisplay(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setText("")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.widget = QtWidgets.QWidget(self.frame)
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
        self.resetCurrentAsparagus = QtWidgets.QPushButton(self.widget_2)
        self.resetCurrentAsparagus.setObjectName("resetCurrentAsparagus")
        self.horizontalLayout.addWidget(self.resetCurrentAsparagus)
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
        self.horizontalLayout_3.addWidget(self.frame)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMinimumSize(QtCore.QSize(200, 0))
        self.scrollArea.setMaximumSize(QtCore.QSize(300, 16777215))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 298, 960))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.groupBoxViolet = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBoxViolet.setObjectName("groupBoxViolet")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBoxViolet)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.color = ImageDisplay(self.groupBoxViolet)
        self.color.setMinimumSize(QtCore.QSize(250, 150))
        self.color.setText("")
        self.color.setObjectName("color")
        self.verticalLayout_5.addWidget(self.color)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.predictionViolet_2 = QtWidgets.QLineEdit(self.groupBoxViolet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionViolet_2.sizePolicy().hasHeightForWidth())
        self.predictionViolet_2.setSizePolicy(sizePolicy)
        self.predictionViolet_2.setMinimumSize(QtCore.QSize(10, 0))
        self.predictionViolet_2.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionViolet_2.setObjectName("predictionViolet_2")
        self.horizontalLayout_5.addWidget(self.predictionViolet_2)
        self.predictionViolet = QtWidgets.QLineEdit(self.groupBoxViolet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionViolet.sizePolicy().hasHeightForWidth())
        self.predictionViolet.setSizePolicy(sizePolicy)
        self.predictionViolet.setMinimumSize(QtCore.QSize(10, 0))
        self.predictionViolet.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionViolet.setObjectName("predictionViolet")
        self.horizontalLayout_5.addWidget(self.predictionViolet)
        self.predictionViolet_3 = QtWidgets.QLineEdit(self.groupBoxViolet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionViolet_3.sizePolicy().hasHeightForWidth())
        self.predictionViolet_3.setSizePolicy(sizePolicy)
        self.predictionViolet_3.setMinimumSize(QtCore.QSize(10, 0))
        self.predictionViolet_3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionViolet_3.setObjectName("predictionViolet_3")
        self.horizontalLayout_5.addWidget(self.predictionViolet_3)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.overallPredictionViolet = QtWidgets.QLineEdit(self.groupBoxViolet)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overallPredictionViolet.sizePolicy().hasHeightForWidth())
        self.overallPredictionViolet.setSizePolicy(sizePolicy)
        self.overallPredictionViolet.setObjectName("overallPredictionViolet")
        self.verticalLayout_5.addWidget(self.overallPredictionViolet)
        self.usePredictionViolet = QtWidgets.QCheckBox(self.groupBoxViolet)
        self.usePredictionViolet.setObjectName("usePredictionViolet")
        self.verticalLayout_5.addWidget(self.usePredictionViolet)
        self.overallPredictionViolet.raise_()
        self.usePredictionViolet.raise_()
        self.color.raise_()
        self.verticalLayout_10.addWidget(self.groupBoxViolet)
        self.groupBoxBlooming = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBoxBlooming.setObjectName("groupBoxBlooming")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBoxBlooming)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.predictionBlooming_2 = QtWidgets.QLineEdit(self.groupBoxBlooming)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionBlooming_2.sizePolicy().hasHeightForWidth())
        self.predictionBlooming_2.setSizePolicy(sizePolicy)
        self.predictionBlooming_2.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionBlooming_2.setObjectName("predictionBlooming_2")
        self.horizontalLayout_11.addWidget(self.predictionBlooming_2)
        self.predictionBlooming = QtWidgets.QLineEdit(self.groupBoxBlooming)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionBlooming.sizePolicy().hasHeightForWidth())
        self.predictionBlooming.setSizePolicy(sizePolicy)
        self.predictionBlooming.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionBlooming.setObjectName("predictionBlooming")
        self.horizontalLayout_11.addWidget(self.predictionBlooming)
        self.predictionBlooming_3 = QtWidgets.QLineEdit(self.groupBoxBlooming)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionBlooming_3.sizePolicy().hasHeightForWidth())
        self.predictionBlooming_3.setSizePolicy(sizePolicy)
        self.predictionBlooming_3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionBlooming_3.setObjectName("predictionBlooming_3")
        self.horizontalLayout_11.addWidget(self.predictionBlooming_3)
        self.verticalLayout_9.addLayout(self.horizontalLayout_11)
        self.overallPredictionBlooming = QtWidgets.QLineEdit(self.groupBoxBlooming)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overallPredictionBlooming.sizePolicy().hasHeightForWidth())
        self.overallPredictionBlooming.setSizePolicy(sizePolicy)
        self.overallPredictionBlooming.setObjectName("overallPredictionBlooming")
        self.verticalLayout_9.addWidget(self.overallPredictionBlooming)
        self.usePredictedValueBlooming = QtWidgets.QCheckBox(self.groupBoxBlooming)
        self.usePredictedValueBlooming.setObjectName("usePredictedValueBlooming")
        self.verticalLayout_9.addWidget(self.usePredictedValueBlooming)
        self.verticalLayout_10.addWidget(self.groupBoxBlooming)
        self.groupBoxLength = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBoxLength.setObjectName("groupBoxLength")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBoxLength)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.predictionLength = QtWidgets.QLineEdit(self.groupBoxLength)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionLength.sizePolicy().hasHeightForWidth())
        self.predictionLength.setSizePolicy(sizePolicy)
        self.predictionLength.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionLength.setObjectName("predictionLength")
        self.horizontalLayout_9.addWidget(self.predictionLength)
        self.predictionLength_2 = QtWidgets.QLineEdit(self.groupBoxLength)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionLength_2.sizePolicy().hasHeightForWidth())
        self.predictionLength_2.setSizePolicy(sizePolicy)
        self.predictionLength_2.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionLength_2.setObjectName("predictionLength_2")
        self.horizontalLayout_9.addWidget(self.predictionLength_2)
        self.predictionLength_3 = QtWidgets.QLineEdit(self.groupBoxLength)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionLength_3.sizePolicy().hasHeightForWidth())
        self.predictionLength_3.setSizePolicy(sizePolicy)
        self.predictionLength_3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionLength_3.setObjectName("predictionLength_3")
        self.horizontalLayout_9.addWidget(self.predictionLength_3)
        self.verticalLayout_7.addLayout(self.horizontalLayout_9)
        self.overallPredictionLength = QtWidgets.QLineEdit(self.groupBoxLength)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overallPredictionLength.sizePolicy().hasHeightForWidth())
        self.overallPredictionLength.setSizePolicy(sizePolicy)
        self.overallPredictionLength.setObjectName("overallPredictionLength")
        self.verticalLayout_7.addWidget(self.overallPredictionLength)
        self.usePredictedValueLength = QtWidgets.QCheckBox(self.groupBoxLength)
        self.usePredictedValueLength.setObjectName("usePredictedValueLength")
        self.verticalLayout_7.addWidget(self.usePredictedValueLength)
        self.verticalLayout_10.addWidget(self.groupBoxLength)
        self.groupBoxRust = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBoxRust.setObjectName("groupBoxRust")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBoxRust)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.predictionRust_2 = QtWidgets.QLineEdit(self.groupBoxRust)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionRust_2.sizePolicy().hasHeightForWidth())
        self.predictionRust_2.setSizePolicy(sizePolicy)
        self.predictionRust_2.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionRust_2.setObjectName("predictionRust_2")
        self.horizontalLayout_7.addWidget(self.predictionRust_2)
        self.predictionRust = QtWidgets.QLineEdit(self.groupBoxRust)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionRust.sizePolicy().hasHeightForWidth())
        self.predictionRust.setSizePolicy(sizePolicy)
        self.predictionRust.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionRust.setObjectName("predictionRust")
        self.horizontalLayout_7.addWidget(self.predictionRust)
        self.predictionRust_3 = QtWidgets.QLineEdit(self.groupBoxRust)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionRust_3.sizePolicy().hasHeightForWidth())
        self.predictionRust_3.setSizePolicy(sizePolicy)
        self.predictionRust_3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionRust_3.setObjectName("predictionRust_3")
        self.horizontalLayout_7.addWidget(self.predictionRust_3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_7)
        self.overallPredictedValueRust = QtWidgets.QLineEdit(self.groupBoxRust)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overallPredictedValueRust.sizePolicy().hasHeightForWidth())
        self.overallPredictedValueRust.setSizePolicy(sizePolicy)
        self.overallPredictedValueRust.setObjectName("overallPredictedValueRust")
        self.verticalLayout_4.addWidget(self.overallPredictedValueRust)
        self.usePredictionRust = QtWidgets.QCheckBox(self.groupBoxRust)
        self.usePredictionRust.setObjectName("usePredictionRust")
        self.verticalLayout_4.addWidget(self.usePredictionRust)
        self.verticalLayout_10.addWidget(self.groupBoxRust)
        self.groupBox = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.predictionWidth = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionWidth.sizePolicy().hasHeightForWidth())
        self.predictionWidth.setSizePolicy(sizePolicy)
        self.predictionWidth.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionWidth.setObjectName("predictionWidth")
        self.horizontalLayout_8.addWidget(self.predictionWidth)
        self.predictionWidth_2 = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionWidth_2.sizePolicy().hasHeightForWidth())
        self.predictionWidth_2.setSizePolicy(sizePolicy)
        self.predictionWidth_2.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionWidth_2.setObjectName("predictionWidth_2")
        self.horizontalLayout_8.addWidget(self.predictionWidth_2)
        self.predictionWidth_3 = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionWidth_3.sizePolicy().hasHeightForWidth())
        self.predictionWidth_3.setSizePolicy(sizePolicy)
        self.predictionWidth_3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionWidth_3.setObjectName("predictionWidth_3")
        self.horizontalLayout_8.addWidget(self.predictionWidth_3)
        self.verticalLayout_6.addLayout(self.horizontalLayout_8)
        self.overallPredictionWidth = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overallPredictionWidth.sizePolicy().hasHeightForWidth())
        self.overallPredictionWidth.setSizePolicy(sizePolicy)
        self.overallPredictionWidth.setObjectName("overallPredictionWidth")
        self.verticalLayout_6.addWidget(self.overallPredictionWidth)
        self.usePredictedValueWidth = QtWidgets.QCheckBox(self.groupBox)
        self.usePredictedValueWidth.setObjectName("usePredictedValueWidth")
        self.verticalLayout_6.addWidget(self.usePredictedValueWidth)
        self.verticalLayout_10.addWidget(self.groupBox)
        self.groupBoxBended = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBoxBended.setObjectName("groupBoxBended")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBoxBended)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.predictionBended_3 = QtWidgets.QLineEdit(self.groupBoxBended)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionBended_3.sizePolicy().hasHeightForWidth())
        self.predictionBended_3.setSizePolicy(sizePolicy)
        self.predictionBended_3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionBended_3.setObjectName("predictionBended_3")
        self.horizontalLayout_10.addWidget(self.predictionBended_3)
        self.predictionBended_2 = QtWidgets.QLineEdit(self.groupBoxBended)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionBended_2.sizePolicy().hasHeightForWidth())
        self.predictionBended_2.setSizePolicy(sizePolicy)
        self.predictionBended_2.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionBended_2.setObjectName("predictionBended_2")
        self.horizontalLayout_10.addWidget(self.predictionBended_2)
        self.predictionBended = QtWidgets.QLineEdit(self.groupBoxBended)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictionBended.sizePolicy().hasHeightForWidth())
        self.predictionBended.setSizePolicy(sizePolicy)
        self.predictionBended.setMaximumSize(QtCore.QSize(80, 16777215))
        self.predictionBended.setObjectName("predictionBended")
        self.horizontalLayout_10.addWidget(self.predictionBended)
        self.verticalLayout_8.addLayout(self.horizontalLayout_10)
        self.overallPredictionBended = QtWidgets.QLineEdit(self.groupBoxBended)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.overallPredictionBended.sizePolicy().hasHeightForWidth())
        self.overallPredictionBended.setSizePolicy(sizePolicy)
        self.overallPredictionBended.setObjectName("overallPredictionBended")
        self.verticalLayout_8.addWidget(self.overallPredictionBended)
        self.usePredictedValueBended = QtWidgets.QCheckBox(self.groupBoxBended)
        self.usePredictedValueBended.setObjectName("usePredictedValueBended")
        self.verticalLayout_8.addWidget(self.usePredictedValueBended)
        self.verticalLayout_10.addWidget(self.groupBoxBended)
        self.groupBoxViolet.raise_()
        self.groupBoxRust.raise_()
        self.groupBox.raise_()
        self.groupBoxLength.raise_()
        self.groupBoxBended.raise_()
        self.groupBoxBlooming.raise_()
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout_3.addWidget(self.scrollArea)
        LabelDialog.setCentralWidget(self.centralwidget)

        self.retranslateUi(LabelDialog)
        QtCore.QMetaObject.connectSlotsByName(LabelDialog)

    def retranslateUi(self, LabelDialog):
        _translate = QtCore.QCoreApplication.translate
        LabelDialog.setWindowTitle(_translate("LabelDialog", "Hand Label Assistant"))
        self.resetCurrentAsparagus.setText(_translate("LabelDialog", "Reset current asparagus"))
        self.previous_question.setText(_translate("LabelDialog", "<<"))
        self.next_question.setText(_translate("LabelDialog", ">>"))
        self.yes.setText(_translate("LabelDialog", "Yes"))
        self.no.setText(_translate("LabelDialog", "No"))
        self.groupBoxViolet.setTitle(_translate("LabelDialog", "Violet"))
        self.usePredictionViolet.setText(_translate("LabelDialog", "Use predicted value"))
        self.groupBoxBlooming.setTitle(_translate("LabelDialog", "Blooming"))
        self.usePredictedValueBlooming.setText(_translate("LabelDialog", "Use predicted value"))
        self.groupBoxLength.setTitle(_translate("LabelDialog", "Length"))
        self.usePredictedValueLength.setText(_translate("LabelDialog", "Use predicted value"))
        self.groupBoxRust.setTitle(_translate("LabelDialog", "Rust"))
        self.usePredictionRust.setText(_translate("LabelDialog", "Use predicted value"))
        self.groupBox.setTitle(_translate("LabelDialog", "Width"))
        self.usePredictedValueWidth.setText(_translate("LabelDialog", "Use predicted value"))
        self.groupBoxBended.setTitle(_translate("LabelDialog", "Bended"))
        self.usePredictedValueBended.setText(_translate("LabelDialog", "Use predicted value"))
from imagedisplay import ImageDisplay
