from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import os
import sys

import traceback

from main_ui import *
from label_dialog_ui import *
import numpy as np

import numpy as np
import pandas as pd
import imageio

import re
from feature_extraction.feature_extraction import *
from feature_extraction.color_plot import color_plot

#from collections import Counter
from pandas_model import PandasModel

from label_dialog import *

class MainApp(QWidget):
    def __init__(self, widget_handled, ui):
        """ Initializes Main App.
        args:
            widget_handled: For this widget events are handeled by MainApp
            ui: User interface generated by pyuic5 (view)
        """
        QWidget.__init__(self, widget_handled)

        self.ui = ui
        self.widget_handled = widget_handled #Widget for event Handling. All events are checked and if not arrow keys passed on. See eventFilter below
        self.file_loader = MainApp.FileLoader(self)

        self.idx_image = 0
        self.images = []
        self.table_model = None
        self.ui.table.setFocusPolicy(Qt.NoFocus)

        self.make_connections()

        self.draw_asparagus()
        self.update_info()

    def make_connections(self):
        """ Establishes connections between user interface elements and functionalities """
        self.ui.next_asparagus.clicked.connect(self.next_image)
        self.ui.next_asparagus.clicked.connect(self.update_info)
        self.ui.previous_asparagus.clicked.connect(self.previous_image)
        self.ui.previous_asparagus.clicked.connect(self.update_info)

        self.file_loader.loaded_files.connect(self.set_filenames)
        self.file_loader.images.connect(self.set_images)
        self.file_loader.idx.connect(self.set_idx)
        self.file_loader.idx.connect(lambda: self.draw_asparagus())
        self.file_loader.idx.connect(lambda: self.update_info())
        self.file_loader.info.connect(lambda x: (self.ui.loading_info.setText(x),print(x)))#self.ui.label_2.setText)


    def set_filenames(self, files):
        """ Sets attribute self.files and draws asparagus piece
        Args:
            files: list of filenames
        """
        self.files = files#All files in subtree

    def set_idx(self, idx):
        """ Sets index for current asparagus piece
        Args:
            idx = idx
        """
        self.idx_image = idx

    def set_images(self, nested_list_of_filenames):
        """ Sets nested list of image filepaths depicting the asparagus piece from all three directions
        Args:
            nested_list_of_filenames = nested list of filenames
        """
        self.images = nested_list_of_filenames


    class FileLoader(QThread):
        loaded_files = pyqtSignal(list)
        images = pyqtSignal(dict)
        idx = pyqtSignal(int)
        info = pyqtSignal(str)
        def __init__(self, outer):
            """ Initializes the file loader
            Args:
                outer: The inctance of the outer class
            """
            super(MainApp.FileLoader, self).__init__()
            self.outer = outer
            self.ui = outer.ui
            self.directory = ""
            self.loading_stage = ""#For reporting via self.info
            self.n_files_found = 0

            self.files = []

        def rek_get_files(self, path, regex):
            """ Gets files that match regex for the specified path and all subdirectories recursively.
            Args:
                path: A path to the parent directory.S
                regex: A regex that specifies the name of files searched for.

            """
            for f in os.scandir(path):
                if f.is_dir():
                    print(".",end="")
                    self.rek_get_files(f.path, regex)
                else:
                    self.n_files_found += 1
                    if (self.n_files_found %10) == 0:
                        self.info.emit(self.loading_stage + str(self.n_files_found) + " files found")
                    if re.match(regex,f.name):
                        self.files.append(path+"/"+f.name)

        def set_filenames(self, directory):
            """ Gets the filenames for the given directory, starts the thread.
            Args:
                names: List of filenames
            """
            self.directory = directory
            self.start()

        def run(self):
            """ Gets attribute filenames and draws asparagus.
                Emits loaded files, images the idx and an update info regarding the progress of loading."""
            self.files = []
            images = []
            self.loading_stage = "Listing names: "
            try:
                self.rek_get_files(self.directory,".*\.png")#Traverse subdirectories & get all .bmp filepaths
            except Exception as e:
                images = []
                idx_image = 0
                return
            ids_to_files = {}

            self.loading_stage = "Sorting: "
            for ci, path in enumerate(self.files):
                if (ci % 100) == 0:
                    self.info.emit(self.loading_stage + " File "+str(ci))


                match = re.search(".*/(.*)_[a-z]\.png",path)
                if match:#Get id using regex.
                    id = int(match.groups()[0])
                    try:#Create list as key of id_to_files if it doesn't exist already.
                        ids_to_files[id]
                    except:
                        ids_to_files[id] = []
                    ids_to_files[id].append(path)#Append filename to list.

            idx = np.min(list(ids_to_files.keys()))

            self.loaded_files.emit(self.files)
            self.images.emit(ids_to_files)
            self.idx.emit(idx)

    def set_label_file(self, path):
        """ Sets output file.
            Args:
                path
        """
        self.label_file = path
        self.load_labels()

    def load_labels(self):
        """ Loads labels from the filepath that is specified by self.label_file. Sets self.labels to be None if loading the file failed """
        try:
            recovered = pd.read_csv(self.label_file, index_col=0, sep =";")
            self.table_model = PandasModel(recovered)
            self.ui.table.setModel(self.table_model)
            self.labels = recovered
        except FileNotFoundError:
            self.labels = None
        except Exception as e:
            print(e)

    def next_image(self):
        """ Updates index to next aspargus and elicits redrawing """
        self.idx_image += 1

        self.draw_asparagus()
        self.update_info()

    def update_info(self):
        """ Updates the infobox such that it contains the filenames of all three images/perspectives displayed at a time"""
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
        except Exception as e:
            print(e)
            return

    def previous_image(self):
        """ Updates index to previous Aspargus"""
        self.idx_image -= 1
        self.draw_asparagus()
        self.update_info()

    def eventFilter(self, source, event):
        """ Filters key events such that arrow keys may be handled.
            Args:
                source: Source of event
                event: Event to be handled
        """
        if event.type() == QtCore.QEvent.KeyRelease:
            id_right = 16777236
            id_left = 16777234

            if event.key() == id_right:
                self.next_image()

                self.draw_asparagus()

            elif event.key() == id_left:
                self.previous_image()

                self.draw_asparagus()

        try:#When closing the app the widget handled might already have been destroyed
            return self.widget_handled.eventFilter(source, event)#Execute the default actions for the event
        except:
            return True#a true value prevents the event from being sent on to other objects



class SourceDirOpener(QWidget):
    """ A class with wich the user may open the source directory containing images that satisfy the naming convention"""
    filenames = pyqtSignal(str)
    def __init__(self):
        """ Initialized the class"""
        QWidget.__init__(self)
    def get_filenames(self):
        """ Opens the file dialog and emits the signal filenames"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(None,"Select folder...",os.getcwd(),options=options)
        if path:
            self.filenames.emit(path)

class OutputFileSelector(QWidget):
    """ A class with wich the user may select an output csv file"""
    outfilepath = pyqtSignal(str)
    def __init__(self):
        """ Initialized the class"""
        QWidget.__init__(self)

    def get_outputfile(self):
        """ Opens the file dialog for existing files and emits the signal outfilepath"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None,"Select the output file", "", "csv (*.csv);;", options=options)
        if fileName:
            self.outfilepath.emit(fileName)

class OutputFileCreator(QWidget):
    """ A class with wich the user create an output csv file"""
    outfilepath = pyqtSignal(str)
    def __init__(self):
        """ Initialized the class"""
        QWidget.__init__(self)

    def create_outputfile(self):
        """ Opens the file dialog for non existing files and emits the signal outfilepath"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None,"Select the output file", "", "csv (*.csv);;", options=options)

        if filename:
            if not filename.endswith(".csv"):
                filename += ".csv"
            self.outfilepath.emit(filename)

class HandLabelAssistant():
    def __init__(self):
        """ Initializes the hand label assistant.
            Starts new pyqt app and creates instances of the views (ui) and the controllers (including the model interfaces; MainApp, LabelingDialog). """
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
        """ Establishes connections between the ui elements that are relevant for windows other then the one handled by the respective controller"""
        # Connect actions (Dropdown menu in upper bar) to file dialogs and file dialog)
        self.ui.actionOpen_labeling_dialog.triggered.connect(self.open_labeling_dialog)#open upon user input
        self.ui.start_labeling.clicked.connect(self.open_labeling_dialog)#open upon user input
        self.source_dir_opener.filenames.connect(self.main_app.file_loader.set_filenames)

        self.main_app.file_loader.images.connect(self.labeling_app.set_images)
        self.main_app.file_loader.idx.connect(self.labeling_app.set_index)

        #self.main_app.file_loader.idx.connect(lambda: self.labeling_app.draw_asparagus())
        #self.main_app.file_loader.idx.connect(lambda: self.labeling_app.update_info())

        self.ui.actionOpen_file_directory.triggered.connect(self.source_dir_opener.get_filenames)

        self.output_file_selector.outfilepath.connect(self.labeling_app.set_output_file)
        self.output_file_selector.outfilepath.connect(self.main_app.set_label_file)
        self.output_file_creator.outfilepath.connect(self.labeling_app.set_output_file)

        self.ui.actionLoad_label_file.triggered.connect(self.output_file_selector.get_outputfile)
        self.ui.actionHelp.triggered.connect(self.print_usage)
        self.ui.actionCreate_new_label_file.triggered.connect(self.output_file_creator.create_outputfile)
        self.ui.actionClose_3.triggered.connect(lambda x: sys.exit())# We close upon click on action

    def print_usage(self):
        """ Opens a message box that contains advice of how to use the app"""
        QMessageBox.about(self.main_app,"Usage", "Specify outputfile and the folder containg valid files first!"
                                                + "\n\n Valid files:"
                                                + "\n\n Valid files are png files named [idx]_a.png, [idx]_b.png and [idx]_c.png where [idx] refers to any integer starting at zero."
                                                + " Named indices are used as the indentifier of each asparagus piece."
                                                + " The PNGs may be contained in subfolders of the outputfolder .However for you to be able to skip through files make sure the indices are in a continuous range (1,2 ... n)"
                                                + "\n\n Output label file:"
                                                + "\n\n The generated labels are saved in a .csv file. The file index is used as the identifier: Each row contains information for one asparagus piece.")


    def open_labeling_dialog(self):
        """ Opens the labeling app in a second window """
        if(type(self.labeling_app.labels) == type(None)):
            self.print_usage()
        else:
             self.label_window.show()

if __name__ == "__main__":
    assistant = HandLabelAssistant()
