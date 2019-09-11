from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import os
import sys

from qevent_to_name import *

from view import *
from label_dialog_ui import *
import numpy as np

import numpy as np
import pandas as pd
import imageio

import re

class MainApp(QWidget):
    coordinates = pyqtSignal(QRect)

    def __init__(self, widget_handled, ui):
        """ Initializes Main App.
        args:
            widget_handled: For this widget events are handeled by MainApp
            ui: User interface
        """
        self.ui = ui
        self.make_connections()
        self.widget_handled = widget_handled #Widget for event Handling. All events are checked and if not arrow keys passed on. See eventFilter below
        QWidget.__init__(self, widget_handled)

        self.idx_image = 0
        self.images = []
        self.draw_asparagus()
        self.update_info()

        self.ui.tree.set_white_background()
        self.ui.tree.update(imageio.imread("tree11.bmp"))

        #For self.nodes_to_ui, a dictionary to map names of nodes in the decision tree to the respective ui elements
        self.ui_nodes = [self.ui.is_bruch,self.ui.is_not_bruch,self.ui.has_blume,self.ui.is_blume_dick,self.ui.is_blume_duenn,self.ui.has_no_blume,self.ui.has_rost,self.ui.is_rost_dick,
                    self.ui.is_rost_duenn,self.ui.has_no_rost,self.ui.is_bended,self.ui.is_krumm_dick,self.ui.is_bended_medium,self.ui.is_bended_medium_non_violet,self.ui.is_bended_medium_violet,self.ui.is_not_bended
                    ,self.ui.is_not_bended_non_violet,self.ui.is_dicke,self.ui.is_anna,self.ui.is_bona,self.ui.is_clara,self.ui.is_suppe,self.ui.is_not_bended_violet,self.ui.is_violet_dick,self.ui.is_violet_duenn]

        self.nodes = ["is_bruch","is_not_bruch","has_blume","is_blume_dick","is_blume_duenn","has_no_blume","has_rost","is_rost_dick",
                    "is_rost_duenn","has_no_rost","is_bended","is_krumm_dick","is_bended_medium","is_bended_medium_non_violet","is_bended_medium_violet","is_not_bended"
                    ,"is_not_bended_non_violet","is_dicke","is_anna","is_bona","is_clara","is_suppe","is_not_bended_violet","is_violet_dick","is_violet_duenn"]

        self.label_file = None
        self.labels = None

        self.nodes_to_ui = {}
        for node, ui_node in zip(self.nodes,self.ui_nodes):
            self.nodes_to_ui[node] = ui_node

    def update_checkboxes(self):
        """ Updates checkboxes"""
        for k,v in self.nodes_to_ui.items():
            self.nodes_to_ui[k].setChecked(False)
            self.nodes_to_ui[k].setEnabled(False)
            self.nodes_to_ui[k].setFocusPolicy(QtCore.Qt.NoFocus)#No focus  via arrow keys

        if type(self.labels)==type(None):
            return

        try:
            categories = ["is_bruch","has_keule","has_blume","has_rost","is_bended","is_violet","very_thick","thick","medium_thick","thin","very_thin"]
            self.current_data = {}
            for integer, paraphrase in zip(self.labels[self.idx_image],categories):
                if integer:
                    self.current_data[paraphrase] = True
                else:
                    self.current_data[paraphrase] = False
        except KeyError:
            return#no info for aspargus piece
        try:
	        if self.current_data["is_bruch"]:
	            self.nodes_to_ui["is_bruch"].setChecked(True)
	            self.nodes_to_ui["is_bruch"].setEnabled(True)
	        else:
	            self.nodes_to_ui["is_not_bruch"].setChecked(True)
	            self.nodes_to_ui["is_not_bruch"].setEnabled(True)

	            if self.current_data["has_blume"]:
	                self.nodes_to_ui["has_blume"].setChecked(True)
	                self.nodes_to_ui["has_blume"].setEnabled(True)

	                if self.current_data["thick"] or self.current_data["very_thick"]:#>20mm
	                    self.nodes_to_ui["is_blume_dick"].setChecked(True)
	                    self.nodes_to_ui["is_blume_dick"].setEnabled(True)
	                else:
	                    self.nodes_to_ui["is_blume_duenn"].setChecked(True)
	                    self.nodes_to_ui["is_blume_duenn"].setEnabled(True)

	            else:
	                self.nodes_to_ui["has_no_blume"].setChecked(True)
	                self.nodes_to_ui["has_no_blume"].setEnabled(True)

	                if self.current_data["has_rost"]:
	                    self.nodes_to_ui["has_rost"].setChecked(True)
	                    self.nodes_to_ui["has_rost"].setEnabled(True)

	                    if self.current_data["thick"] or self.current_data["very_thick"]:#>20mm
	                        self.nodes_to_ui["is_rost_dick"].setChecked(True)
	                        self.nodes_to_ui["is_rost_dick"].setEnabled(True)
	                    else:
	                        self.nodes_to_ui["is_rost_duenn"].setChecked(True)
	                        self.nodes_to_ui["is_rost_duenn"].setEnabled(True)
	                else:
	                    self.nodes_to_ui["has_no_rost"].setChecked(True)
	                    self.nodes_to_ui["has_no_rost"].setEnabled(True)

	                    if self.current_data["is_bended"]:
	                        self.nodes_to_ui["is_bended"].setChecked(True)
	                        self.nodes_to_ui["is_bended"].setEnabled(True)
	                        if self.current_data["very_thick"]:#>26mm
	                            self.nodes_to_ui["is_krumm_dick"].setChecked(True)
	                            self.nodes_to_ui["is_krumm_dick"].setEnabled(True)
	                        else:
	                            self.nodes_to_ui["is_bended_medium"].setChecked(True)
	                            self.nodes_to_ui["is_bended_medium"].setEnabled(True)

	                            if self.current_data["is_violet"]:
	                                self.nodes_to_ui["is_bended_medium_violet"].setChecked(True)
	                                self.nodes_to_ui["is_bended_medium_violet"].setEnabled(True)
	                            else:
	                                self.nodes_to_ui["is_bended_medium_non_violet"].setChecked(True)
	                                self.nodes_to_ui["is_bended_medium_non_violet"].setEnabled(True)
	                    else:
	                        self.nodes_to_ui["is_not_bended"].setChecked(True)
	                        self.nodes_to_ui["is_not_bended"].setEnabled(True)
	                        if self.current_data["is_violet"]:
	                                self.nodes_to_ui["is_not_bended_violet"].setChecked(True)
	                                self.nodes_to_ui["is_not_bended_violet"].setEnabled(True)

	                                if self.current_data["thick"] or self.current_data["very_thick"]:#>20mm
	                                    self.nodes_to_ui["is_violet_dick"].setChecked(True)
	                                    self.nodes_to_ui["is_violet_dick"].setEnabled(True)
	                                else:
	                                    self.nodes_to_ui["is_violet_duenn"].setChecked(True)
	                                    self.nodes_to_ui["is_violet_duenn"].setEnabled(True)
	                        else:
	                                self.nodes_to_ui["is_not_bended_non_violet"].setChecked(True)
	                                self.nodes_to_ui["is_not_bended_non_violet"].setEnabled(True)
	                                if self.current_data["very_thick"]:
	                                    self.nodes_to_ui["is_dicke"].setChecked(True)
	                                    self.nodes_to_ui["is_dicke"].setEnabled(True)
	                                elif self.current_data["thick"]:
	                                    self.nodes_to_ui["is_anna"].setChecked(True)
	                                    self.nodes_to_ui["is_anna"].setEnabled(True)
	                                elif self.current_data["medium_thick"]:
	                                    self.nodes_to_ui["is_bona"].setChecked(True)
	                                    self.nodes_to_ui["is_bona"].setEnabled(True)
	                                elif self.current_data["thin"]:
	                                    self.nodes_to_ui["is_clara"].setChecked(True)
	                                    self.nodes_to_ui["is_clara"].setEnabled(True)
	                                elif self.current_data["very_thin"]:
	                                    self.nodes_to_ui["is_suppe"].setChecked(True)
	                                    self.nodes_to_ui["is_suppe"].setEnabled(True)
	                                else:
	                                    print("Missing thickness value!!!")
        except:
            print("Invalid file!!!")


    def make_connections(self):
        """ Establishes connections between user interface elements and functionalities"""
        self.ui.next_asparagus.clicked.connect(self.next_image)
        self.ui.next_asparagus.clicked.connect(self.update_info)
        self.ui.previous_asparagus.clicked.connect(self.previous_image)
        self.ui.previous_asparagus.clicked.connect(self.update_info)

    def set_label_file(self, path):
        """ Sets output file.
            Args:
                path
        """
        print(path)
        self.label_file = path
        self.load_labels()

    def load_labels(self):
        try:
            recovered = pd.read_csv(self.label_file, index_col=0, sep =";").to_dict(orient="index")
            for key, value in recovered.items():
                recovered[key] = list(recovered[key].values())
            self.labels = recovered
        except FileNotFoundError:
            self.labels = {}
        except Exception as e:
            print(e)
        self.update_checkboxes()

    def rek_get_files(self, path, regex):
        for f in os.scandir(path):
            if f.is_dir():
                self.rek_get_files(f.path, regex)
            else:
                if re.match(regex,f.name):
                    self.files.append(path+"/"+f.name)

    def set_filenames(self, directory):
        """ Sets attribute filenames and draws asparagus
        Args:
            names: List of filenames
        """
        print(directory)
        self.files = []
        try:
            self.rek_get_files(directory,".*\.png")#Traverse subdirectories & get all .bmp filepaths
        except:
            self.images = []
            self.idx_image = 0
            return
        ids_to_files = {}

        for path in self.files:
            match = re.search(".*/(.*)_[a-z]\.png",path)
            if match:#Get id using regex.
                id = match.groups()[0]
                try:#Create list as key of id_to_files if it doesn't exist already.
                    ids_to_files[id]
                except:
                    ids_to_files[id] = []
                ids_to_files[id].append(path)#Append filename to list.

        self.images = list(ids_to_files.values())
        self.images.sort()

        self.idx_image = 0
        self.draw_asparagus()

    def next_image(self):
        """Updates index to next aspargus and elicits redrawing"""
        if self.idx_image + 1 >= len(self.images):
            return
        self.idx_image += 1

        self.update_checkboxes()
        self.draw_asparagus()
        self.update_info()

    def draw_asparagus(self):
        """ Draws image of asparagus pieces from three perspectives"""
        try:
            imgs = []
            max_y = 0
            max_x = 0
            for i, fname in enumerate(self.images[self.idx_image]):
                im = imageio.imread(fname)
                im = np.rot90(im).copy()
                imgs.append(im)

                max_y += im.shape[0]
                if im.shape[1] > max_x:
                    max_x = im.shape[1]


            n_channels = imgs[0].shape[2]
            combined = np.zeros([max_y,max_x,n_channels],dtype=np.uint8)
            y_offset = 0
            for im in imgs:
                combined[y_offset:y_offset+im.shape[0],:im.shape[1],:] = im
                y_offset += im.shape[0]

            self.ui.label.update(combined)
        except:
            return

    def update_info(self):
        try:
            msg = ""
            for img_path in self.images[self.idx_image]: # change to idx??
                img_name = re.search(".*/(.*_[a-z]\.png)",img_path).groups()[0]
                msg += " " + img_name + " "
            self.ui.label_2.setText(msg)
        except:
            pass

    def draw_asparagus(self):
        """ Draws image of asparagus pieces from three perspectives"""
        try:
            imgs = []
            max_y = 0
            max_x = 0
            for i, fname in enumerate(self.images[self.idx_image]):
                im = imageio.imread(fname)
                im = np.rot90(im).copy()
                imgs.append(im)

                max_y += im.shape[0]
                if im.shape[1] > max_x:
                    max_x = im.shape[1]


            n_channels = imgs[0].shape[2]
            combined = np.zeros([max_y,max_x,n_channels],dtype=np.uint8)
            y_offset = 0
            for im in imgs:
                combined[y_offset:y_offset+im.shape[0],:im.shape[1],:] = im
                y_offset += im.shape[0]

            self.ui.label.update(combined)
        except:
            return

    def previous_image(self):
        """ Updates index to previous Aspargus"""
        if self.idx_image == 0:
            return

        self.idx_image -= 1
        self.update_checkboxes()
        self.draw_asparagus()
        self.update_info()

    def eventFilter(self, source, event):
        """ Filters key events such that arrow keys may be handled.
            Args:
                source: Source of event
                event: Event to be handled
        """
        if event.type() == QtCore.QEvent.KeyRelease:
            #id_up = 16777235
            id_right = 16777236
            id_left = 16777234

            if event.key() == id_right:
                self.next_image()

                self.draw_asparagus()

            elif event.key() == id_left:
                self.previous_image()

                self.draw_asparagus()


        return self.widget_handled.eventFilter(source, event)#forward event

class LabelingDialog(QWidget):
    coordinates = pyqtSignal(QRect)

    def __init__(self, widget_handled, ui):
        """ Initializes LabelingDialog App
        Args:
            widget_handeled: Events for this widget are handeled by LabelingDialog to access arrow keys
            ui: User interface
        """

        self.ui = ui
        self.make_conncections()
        self.widget_handled = widget_handled #Widget for event Handling. All events are checked and if not arrow keys passed on. See eventFilter below
        QWidget.__init__(self, widget_handled)

        self.outpath = None
        self.labels = None#After loading a dLast image reachedictionary that contains key= index of asparagus to value=list of properties

        self.idx_image = 0
        self.images = {}

        self.questions = ["is_bruch","has_keule","has_blume","has_rost","is_bended","is_violet","very_thick","thick","medium_thick","thin","very_thin"]

        self.idx_question = 0
        self.ui.question.setText(self.questions[self.idx_question])

    def make_conncections(self):
        """ Establish connections between UI elements and functionalities"""
        self.ui.asparagus_no.valueChanged.connect(lambda x: self.set_index(int(x)))

        self.ui.previous_question.clicked.connect(self.previous_question)
        self.ui.next_question.clicked.connect(self.next_question)
        self.ui.yes.clicked.connect(self.answer_yes)
        self.ui.no.clicked.connect(self.answer_no)

    def set_output_file(self, path):
        """ Sets output file
            Args:
                path
        """
        self.outpath = path
        self.load_outfile()

    def load_outfile(self):
        """ Loads outputfile if it exists and saves contents to self.dict.
            Sets self.labels to an empty dict otherwise."""
        try:
            recovered = pd.read_csv(self.outpath, index_col=0, sep =";").to_dict(orient="index")
            for key, value in recovered.items():
                recovered[key] = list(recovered[key].values())
            self.labels = recovered
        except FileNotFoundError:
            self.labels = {}
        except e as Exception:
            print(e)

    def answer_yes(self):
        """ Writes answer (yes) to current question to file """
        self.log_answer(1)
        self.ui.yes.setFocus()
        self.next_question()

    def answer_no(self):
        """ Writes answer (no) to current question to file """
        self.log_answer(0)
        self.ui.no.setFocus()
        self.next_question()

    def log_answer(self, answer):
        assert type(self.labels) == type({})
        try:#Assure we have a line of data for the current index
            self.labels[self.idx_image]
        except KeyError:
            self.labels[self.idx_image] = [None for x in range(len(self.questions))]
        self.labels[self.idx_image][self.idx_question] = answer

    def write_answers_to_file(self):
        """ Writes answer to current question to file """
        df = pd.DataFrame.from_dict(self.labels, orient = "index", columns=self.questions)
        df.to_csv(self.outpath,sep =";")
        #self.outpath
        #if self.idx_image

        #self.label_array[self.idx_image,self.idx_question] = answer

        #pd.DataFrame(self.label_array, columns =self.questions).to_csv(self.outpath,sep =";")
        return

    def previous_question(self):
        """ Changes index to previous file and draws respective asparagus"""
        if(self.idx_question==0):
            self.previous_image()
            self.idx_question = len(self.questions)-1
        else:
            self.idx_question -= 1
        self.ui.question.setText(self.questions[self.idx_question])

    def next_question(self):
        """ Changes index to next file and draws respective asparagus"""
        if(self.idx_question==len(self.questions)-1):
            self.next_image()
            self.idx_question = 0
        else:
            self.idx_question += 1
        self.ui.question.setText(self.questions[self.idx_question])

    def rek_get_files(self, path, regex):
        for f in os.scandir(path):
            if f.is_dir():
                self.rek_get_files(f.path, regex)
            else:
                if re.match(regex,f.name):
                    self.files.append(path+"/"+f.name)

    def set_filenames(self, directory):
        """ Sets attribute filenames and draws asparagus
        Args:
            names: List of filenames
        """

        self.files = []#All files in subtree
        self.rek_get_files(directory,".*\.png")#Traverse subdirectories & get all .bmp filepaths

        ids_to_files = {}

        for path in self.files:
            match = re.search(".*/([0-9]+)_[a-z]\.png",path)
            if match:#Get id using regex.
                id = int(match.groups()[0])
                try:#Create list as key of id_to_files if it doesn't exist already.
                    ids_to_files[id]
                except:
                    ids_to_files[id] = []
                ids_to_files[id].append(path)#Append filename to list.

        self.images = ids_to_files#list(ids_to_files.values())
        #self.images.sort()

        self.idx_image = 0
        self.draw_asparagus()

    def next_image(self):
        self.write_answers_to_file()

        #if self.idx_image + 1 >= len(self.images):
        #    QMessageBox.about(self, "Attention", "Last image reached")
        #    return

        self.idx_image += 1
        self.draw_asparagus()
        self.update_info()
        self.ui.asparagus_no.blockSignals(True)
        self.ui.asparagus_no.setValue(self.idx_image)
        self.ui.asparagus_no.blockSignals(False)

    def draw_asparagus(self):
        """ Draws image of asparagus pieces from three perspectives"""
        try:
            imgs = []
            max_y = 0
            max_x = 0
            for i, fname in enumerate(self.images[self.idx_image]):
                im = imageio.imread(fname)
                im = np.rot90(im).copy()
                imgs.append(im)

                max_y += im.shape[0]
                if im.shape[1] > max_x:
                    max_x = im.shape[1]

            n_channels = imgs[0].shape[2]
            combined = np.zeros([max_y,max_x,n_channels],dtype=np.uint8)

            y_offset = 0
            for im in imgs:
                combined[y_offset:y_offset+im.shape[0],:im.shape[1],:] = im
                y_offset += im.shape[0]

            self.ui.label.update(combined)
        except KeyError:
            QMessageBox.about(self, "Attention", "No images found for current index (" + str(self.idx_image) + ")")
        except Exception as e:
            print(e)
            return

    def update_info(self):
        try:
            msg = ""
            for img_path in self.images[self.idx_image]: # change to idx??
                img_name = re.search(".*/(.*_[a-z]\.png)",img_path).groups()[0]
                msg += " // " + img_name + " // "
            self.ui.asparagus_name.setText(msg)
        except:
            pass

    def previous_image(self):
        if self.idx_image -1 < 0:
            QMessageBox.about(self, "Attention", "First image reached")
            return
        self.idx_image -= 1
        self.draw_asparagus()
        self.update_info()
        self.ui.asparagus_no.blockSignals(True)
        self.ui.asparagus_no.setValue(self.idx_image)
        self.ui.asparagus_no.blockSignals(False)

    def set_index(self,idx):
        self.idx_image = idx
        self.draw_asparagus()
        self.update_info()
        self.idx_question = 0
        self.ui.question.setText(self.questions[self.idx_question])

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.KeyRelease:
            #id_up = 16777235
            id_right = 16777236
            id_left = 16777234

            if event.key() == id_right:
                self.answer_no()

            elif event.key() == id_left:
                self.answer_yes()

        return self.widget_handled.eventFilter(source, event)#forward event

class SourceDirOpener(QWidget):
    filenames = pyqtSignal(str)
    def __init__(self):
        QWidget.__init__(self)
    def get_filenames(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(None,"Select folder...",os.getcwd(),options=options)
        if path:
            self.filenames.emit(path)

class OutputFileSelector(QWidget):
    outfilepath = pyqtSignal(str)
    def __init__(self):
        QWidget.__init__(self)

    def get_outputfile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None,"Select the output file", "", "csv (*.csv);;", options=options)
        if fileName:
            self.outfilepath.emit(fileName)

class OutputFileCreator(QWidget):
    outfilepath = pyqtSignal(str)
    def __init__(self):
        QWidget.__init__(self)

    def create_outputfile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None,"Select the output file", "", "csv (*.csv);;", options=options)

        if filename:
            if not filename.endswith(".csv"):
                filename += ".csv"
            self.outfilepath.emit(filename)

class HandLabelAssistant():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)

        #[1] First we open our main window and install an event handler/controller
        MainWindow = QtWidgets.QMainWindow()#Create a window
        self.ui = Ui_Asparator()#Instanciate our UI
        self.ui.setupUi(MainWindow)#Setup our UI as this MainWindow

        self.main_app = MainApp(self.ui.centralwidget, self.ui)#Install MainApp as event filter for handling of arrow keys
        # Note: does not capture arrow keys as respective events are used to target on GUI elements:
        # MainWindow.keyPressEvent = lambda e: print(e.key())

        self.ui.centralwidget.installEventFilter(self.main_app)
        MainWindow.showMaximized()#Doesn't work via XMING

        #[2] Then here comes the code to open another window. The interactive labeling interface...
        self.label_window = QtWidgets.QMainWindow(parent=MainWindow)
        ui_label_assistant = Ui_LabelDialog()
        ui_label_assistant.setupUi(self.label_window)
        self.labeling_app = LabelingDialog(ui_label_assistant.centralwidget, ui_label_assistant)#Install MainApp as event filter
        ui_label_assistant.centralwidget.installEventFilter(self.labeling_app)

        self.source_dir_opener = SourceDirOpener()#We open a file dialog upon click on action
        self.output_file_selector = OutputFileSelector()#We open a file dialog upon click on action...
        self.output_file_creator = OutputFileCreator()#We open a file dialog upon click on action...


        self.make_connections()
        MainWindow.show()#and we show it directly
        sys.exit(app.exec_())

    def make_connections(self):
        self.ui.actionOpen_labeling_dialog.triggered.connect(self.open_labeling_dialog)#open upon user input
        # Connect actions (Dropdown menu in upper bar) to file dialogs and file dialogs to methods of
        self.source_dir_opener.filenames.connect(self.main_app.set_filenames)
        self.source_dir_opener.filenames.connect(self.labeling_app.set_filenames)#Set filenames for both apps if a directory is chosen
        self.ui.actionOpen_file_directory.triggered.connect(self.source_dir_opener.get_filenames)

        self.output_file_selector.outfilepath.connect(self.main_app.set_label_file)
        self.output_file_selector.outfilepath.connect(self.labeling_app.set_output_file)

        self.output_file_creator.outfilepath.connect(self.main_app.set_label_file)
        self.output_file_creator.outfilepath.connect(self.labeling_app.set_output_file)

        self.ui.actionLoad_label_file.triggered.connect(self.output_file_selector.get_outputfile)
        self.ui.actionCreate_new_label_file.triggered.connect(self.output_file_creator.create_outputfile)
        self.ui.actionClose_3.triggered.connect(lambda x: sys.exit())# We close upon click on action

    def open_labeling_dialog(self):
        if(type(self.labeling_app.labels) == type(None)):
            QMessageBox.about(self.main_app,"Attention", "Specify outputfile first!")
        else:
             self.label_window.show()

if __name__ == "__main__":
    assistant = HandLabelAssistant()
