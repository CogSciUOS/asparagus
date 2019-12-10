import numpy as np
import os
from PIL import Image
import pickle


class Asparagus:
    def __init__(self, path='/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/without_background_pngs/', batch_size=10, train_test_split=0.8, reload_filenames=False):
        self.batch_size = batch_size
        self.max_idx = 0
        self.files = []
        self.train_test_split = train_test_split

        self.return_validation = False
        self.return_training = True
        self.current_training_batch = 0

        self.init_filenames(path, reload_filenames)

    def init_filenames(self, root, reload_filenames):
        """ Inits filenames. Reads all files contained in subfolders of path.
        params:
            path : directory of folder with subfolders
        """
        files = []
        if not reload_filenames:
            try:
                with open("image_filepaths.pkl", "rb") as infile:
                     files = pickle.load(infile)
            except:
                pass
        if len(files) == 0:
            for gridjob_folder in os.listdir(root):
                print("Loading gridjobfolder")
                if os.path.isfile(gridjob_folder):
                    continue
                for folder in os.listdir(os.path.join(root, gridjob_folder)):
                    print(".", end="")
                    if os.path.isfile(folder):
                        continue
                    full_folder = os.path.join(os.path.join(root, gridjob_folder), folder)
                    for file in os.listdir(full_folder):
                        idx = int(file[:-6])
                        perspective = file[-5]
                        files.append(os.path.join(full_folder, file))
                with open("image_filepaths.pkl", "wb") as outfile:
                    pickle.dump(files, outfile)

        np.random.shuffle(files)
        self.max_idx = len(files)
        self.files = files

    def get_training_batches(self):
        self.return_validation = False
        self.return_training = True
        self.current_training_batch = 0
        return iter(self)

    def get_validation_batches(self):
        self.return_validation = True
        self.return_training = False
        self.current_validation_batch = int(len(self.files)*self.train_test_split/self.batch_size)
        return iter(self)


    def __iter__(self):
        return self

    def __next__(self):
        if self.return_training:
            if self.current_training_batch >= self.max_idx * self.train_test_split:
                self.current_training_batch = 0
            start_idx = (self.current_training_batch+1) * self.batch_size
            stop_idx = start_idx + self.batch_size
            self.current_training_batch += self.batch_size
            return [np.array(Image.open(f)) for f in self.files[start_idx:stop_idx]]
        elif self.return_validation:
            if self.current_validation_batch >= len(self.files):
                self.current_validation_batch = int(len(self.files)*self.train_test_split)
            start_idx = (self.current_validation_batch+1) * self.batch_size
            stop_idx = start_idx + self.batch_size
            self.current_validation_batch += self.batch_size

            #print(self.files[int(len(self.files)*0.8):int(len(self.files)*0.8)+10])
            return [np.array(Image.open(f)) for f in self.files[start_idx:stop_idx]]
