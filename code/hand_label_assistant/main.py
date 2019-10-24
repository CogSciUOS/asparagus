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
from feature_extraction.feature_extraction import *
from feature_extraction.color_plot import color_plot

from collections import Counter

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
	                            self.nodes_feature_extractionto_ui["is_bended_medium"].setEnabled(True)

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
    def __init__(self, widget_handled, ui):
        """ Initializes LabelingDialog App
        Args:
            widget_handeled: Events for this widget are handeled by LabelingDialog to access arrow keys
            ui: User interface
        """
        self.thread = LabelingDialog.Features(self)
        self.extract_features = False
        self.ui = ui
        self.make_conncections()
        self.widget_handled = widget_handled #Widget for event Handling. All events are checked and if not arrow keys passed on. See eventFilter below
        QWidget.__init__(self, widget_handled)

        self.outpath = None
        self.labels = None#After loading a dictionary that contains: key=*index of asparagus* to value=*list of properties*

        self.idx_image = 0
        self.images = {}
        self.ui.color.set_gray_background()


        headers_main_variables = ["is_bruch","is_hollow","has_blume","has_rost_head","has_rost_body","is_bended","is_violet","very_thick","thick","medium_thick","thin","very_thin","unclassified"]
        headers_set_via_feature_extraction = ["auto_violet","auto_blooming","auto_length","auto_rust_head","auto_rust_body","auto_width","auto_bended"]
        headers_additional_extracted_features = []

        self.outfile_headers = []
        self.outfile_headers.extend(headers_main_variables)
        self.outfile_headers.extend(headers_set_via_feature_extraction)
        self.outfile_headers.extend(headers_additional_extracted_features)

        self.questions = headers_main_variables[:-1]#Questions must be the first rows in out


        self.feature_to_questions = { "width":["very_thick","thick","medium_thick","thin","very_thin"],
                                 "blooming":["has_blume"],
                                 "length":["is_bruch"],
                                 "rust":["has_rost_head","has_rost_body"],
                                 "violet":["is_violet"],
                                 "bended":["is_bended"]
                                }
        self.automatic = []

        self.idx_question = 0
        self.ui.question.setText(self.questions[self.idx_question])
        self.update_info()

    def not_classifiable(self):
        try:
            self.set_value_for_label(1, "unclassified")
            self.next_image()
        except e as Exception:
            print(e)

    def use_feature_extraction_for(self, feature):
        remove = self.feature_to_questions[feature]
        for x in remove:
            self.questions.remove(x)

    def use_question_for(self, feature):
        print("I should see you")
        self.questions.extend(self.feature_to_questions[feature])


    def make_conncections(self):
        """ Establish connections between UI elements and functionalities"""
        self.ui.asparagus_number.valueChanged.connect(lambda x: self.set_index(int(x)))
        self.ui.previous_question.clicked.connect(self.previous_question)
        self.ui.next_question.clicked.connect(self.next_question)
        self.ui.yes.clicked.connect(self.answer_yes)
        self.ui.no.clicked.connect(self.answer_no)
        self.ui.notClassifiable.clicked.connect(self.not_classifiable)

        self.ui.usePredictionLength.stateChanged.connect(lambda x: self.use_feature_extraction_for("length") if x else self.use_question_for("length"))
        self.ui.usePredictionBlooming.stateChanged.connect(lambda x: self.use_feature_extraction_for("blooming") if x else self.use_question_for("blooming"))
        self.ui.usePredictionRust.stateChanged.connect(lambda x: self.use_feature_extraction_for("rust") if x else self.use_question_for("rust"))
        self.ui.usePredictionWidth.stateChanged.connect(lambda x: self.use_feature_extraction_for("width") if x else self.use_question_for("width"))
        self.ui.usePredictionBended.stateChanged.connect(lambda x: self.use_feature_extraction_for("bended") if x else self.use_question_for("bended"))
        self.ui.usePredictionViolet.stateChanged.connect(lambda x: self.use_feature_extraction_for("violet") if x else self.use_question_for("violet"))

        self.ui.extractFeatures.toggled.connect(self.toggle_feature_extraction)

        self.thread.color_plot.connect(self.ui.color.update)

        self.thread.predictionWidth.connect(self.ui.predictionWidth.setText)
        self.thread.predictionBended.connect(self.ui.predictionBended.setText)
        self.thread.predictionLength.connect(self.ui.predictionLength.setText)
        self.thread.predictionViolet.connect(self.ui.predictionViolet.setText)
        self.thread.predictionRust.connect(self.ui.predictionRust.setText)
        self.thread.predictionBlooming.connect(self.ui.predictionBlooming.setText)

        self.thread.predictionWidth_2.connect(self.ui.predictionWidth_2.setText)
        self.thread.predictionBended_2.connect(self.ui.predictionBended_2.setText)
        self.thread.predictionLength_2.connect(self.ui.predictionLength_2.setText)
        self.thread.predictionViolet_2.connect(self.ui.predictionViolet_2.setText)
        self.thread.predictionRust_2.connect(self.ui.predictionRust_2.setText)
        self.thread.predictionBlooming_2.connect(self.ui.predictionBlooming_2.setText)

        self.thread.predictionWidth_3.connect(self.ui.predictionWidth_3.setText)
        self.thread.predictionBended_3.connect(self.ui.predictionBended_3.setText)
        self.thread.predictionLength_3.connect(self.ui.predictionLength_3.setText)
        self.thread.predictionViolet_3.connect(self.ui.predictionViolet_3.setText)
        self.thread.predictionRust_3.connect(self.ui.predictionRust_3.setText)
        self.thread.predictionBlooming_3.connect(self.ui.predictionBlooming_3.setText)

        self.thread.overallPredictionWidth.connect(self.ui.overallPredictionWidth.setText)
        self.thread.overallPredictionBended.connect(self.ui.overallPredictionBended.setText)
        self.thread.overallPredictionLength.connect(self.ui.overallPredictionLength.setText)
        self.thread.overallPredictionViolet.connect(self.ui.overallPredictionViolet.setText)
        self.thread.overallPredictedValueRust.connect(self.ui.overallPredictedValueRust.setText)
        self.thread.overallPredictionBlooming.connect(self.ui.overallPredictionBlooming.setText)


    def toggle_feature_extraction(self):
        self.extract_features = not self.extract_features
        self.set_index(self.idx_image)

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

    def set_value_for_label(self, value, label, idx = None):
        """ Sets value for a given label. Exception passed on if self.labels is not set yet."""
        if not type(self.labels) == type({}):
            QMessageBox.about(self, "Attention", "Load output file first.")
            return

        if idx == None:
            idx = self.idx_image

        try:#Assure we have a line of data for the current index
            self.labels[idx]
        except KeyError:
            self.labels[idx] = [None for x in range(len(self.outfile_headers))]

        try:
            self.labels[idx][self.outfile_headers.index(label)] = value
        except e as Exception:
            print("Couldn't set label")
            print(e)

    def log_answer(self, answer):
        self.set_value_for_label(answer,self.questions[self.idx_question])

    def write_answers_to_file(self):
        """ Writes answer to current question to file """
        try:
            df = pd.DataFrame.from_dict(self.labels, orient = "index", columns=self.outfile_headers)
            df.to_csv(self.outpath,sep =";")
        except:
            QMessageBox.about(self, "Attention", "Writing file failed.")


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
        self.idx_image = 0
        self.draw_asparagus()

    def next_image(self):
        self.write_answers_to_file()
        self.set_index(self.idx_image+1)


    class Features(QThread):
        color_plot = pyqtSignal(np.ndarray)
        predictionWidth = pyqtSignal(str)
        predictionBended = pyqtSignal(str)
        predictionLength = pyqtSignal(str)
        predictionViolet = pyqtSignal(str)
        predictionRust = pyqtSignal(str)
        predictionBlooming = pyqtSignal(str)

        predictionWidth_2 = pyqtSignal(str)
        predictionBended_2 = pyqtSignal(str)
        predictionLength_2 = pyqtSignal(str)
        predictionViolet_2 = pyqtSignal(str)
        predictionRust_2 = pyqtSignal(str)
        predictionBlooming_2 = pyqtSignal(str)

        predictionWidth_3 = pyqtSignal(str)
        predictionBended_3 = pyqtSignal(str)
        predictionLength_3 = pyqtSignal(str)
        predictionViolet_3 = pyqtSignal(str)
        predictionRust_3 = pyqtSignal(str)
        predictionBlooming_3 = pyqtSignal(str)

        overallPredictionWidth = pyqtSignal(str)
        overallPredictionBended = pyqtSignal(str)
        overallPredictionLength = pyqtSignal(str)

        overallPredictionViolet = pyqtSignal(str)
        overallPredictedValueRust = pyqtSignal(str)
        overallPredictionBlooming = pyqtSignal(str)

        def __init__(self, outer):
            super().__init__()
            self.outer = outer


        def run(self):
            try:
                imgs = [np.array(imageio.imread(fname)) for fname in self.outer.images[self.outer.idx_image]]
                idx_image = self.outer.idx_image#At creation time
            except KeyError:
                #If there are no images for the index for whatever reason. Just dont do anything.
                self.outer.ui.asparagus_number.setEnabled(True)
                return

            try:
                self.color_plot.emit(color_plot(imgs))
            except:
                self.outer.ui.asparagus_number.setEnabled(True)
                return

            try:
                p = [estimate_width(np.rot90(x)) for x in imgs]
                self.predictionWidth.emit(str(int(p[0][1])))#Numerical widthprint('% 6.2f' % v)
                self.predictionWidth_2.emit(str(int(p[1][1])))
                self.predictionWidth_3.emit(str(int(p[2][1])))
                most_common = Counter(np.array(p)[:,0]).most_common(1)[0][0]
                self.overallPredictionWidth.emit(most_common)
                self.outer.set_value_for_label(most_common, "auto_width",idx_image)
            except Exception as e:
                print(e)

            try:
                p = [estimate_purple(x, threshold_purple=10) for x in imgs]
                self.predictionViolet.emit(str(p[0]))#Numerical widthprint('% 6.2f' % v)
                self.predictionViolet_2.emit(str(p[1]))
                self.predictionViolet_3.emit(str(p[2]))
                most_common = Counter(np.array(p)).most_common(1)[0][0]
                self.overallPredictionViolet.emit(str(most_common))
                self.outer.set_value_for_label(int(most_common), "auto_violet",idx_image)
            except Exception as e:
                print(e)

            try:
                p = [estimate_bended(x,threshold = 120) for x in imgs]
                self.predictionBended.emit(str(int(p[0][1])))#'{:10.1}'.format(p[0][1]))
                self.predictionBended_2.emit(str(int(p[1][1])))#'{:10.1}'.format(p[1][1]))
                self.predictionBended_3.emit(str(int(p[2][1])))#'{:10.1}'.format(p[2][1]))
                is_bended = np.sum(np.array(p)[:,0])>1#If at least one image shows it's bended
                self.overallPredictionBended.emit(str(is_bended))
                self.outer.set_value_for_label(int(is_bended), "auto_bended",idx_image)
            except Exception as e:
                print(e)

            try:
                p = [estimate_length(x) for x in imgs]
                self.predictionLength.emit(str(int(p[0])))
                self.predictionLength_2.emit(str(int(p[1])))
                self.predictionLength_3.emit(str(int(p[2])))
                self.overallPredictionLength.emit(str(np.mean(np.array(p))))
            except Exception as e:
                print(e)

            #ADD YOUR CODE HERE
            self.outer.ui.asparagus_number.setEnabled(True)


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

            self.ui.label.update(np.rot90(combined,3).copy())
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
                msg += " | " + img_name + " | "
            self.ui.asparagus_name.setText(msg)
        except:
            pass

    def previous_image(self):
        self.set_index(self.idx_image-1)


    def set_index(self,idx):
        if(self.thread.isRunning()):
            return
        self.idx_image = idx
        self.draw_asparagus()
        self.update_info()


        self.ui.asparagus_number.blockSignals(True)
        self.ui.asparagus_number.setValue(self.idx_image)
        self.ui.asparagus_number.blockSignals(False)

        self.idx_question = 0
        self.ui.question.setText(self.questions[self.idx_question])


        if self.extract_features:
            self.ui.asparagus_number.setEnabled(False)
            self.thread.start()


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
            self.label_window.show()
        else:
             self.label_window.show()

if __name__ == "__main__":
    assistant = HandLabelAssistant()
