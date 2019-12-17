import numpy as np
import os
from PIL import Image
import pickle

import copy


class Asparagus:
    def __init__(self, path='/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/', batch_size=10, train_test_split=0.8, reload_filenames=False, flatten = False, random_seed = 42):
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.x_train = []
        self.flatten = flatten
        self.x_test = []
        self.return_validation = False
        self.return_training = True
        self.idx = 0
        self.tensor_shape = None

        self.init_filenames(path, reload_filenames, random_seed)

    def init_filenames(self, root, reload_filenames, random_seed):
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

                np.random.seed(seed)
                np.random.shuffle(files)

        train_batches = int((self.train_test_split * len(files))//self.batch_size) #e.g. 1
        start_test = train_batches * self.batch_size+1 #e.g. 11
        self.x_train = files[:start_test]
        self.x_test = files[start_test:]

    def get_training_batches(self):
        self.return_validation = False
        self.return_training = True
        self.idx = 0
        iterator = copy.deepcopy(self)
        return iter(iterator)

    def get_validation_batches(self):
        self.return_validation = True
        self.return_training = False
        self.idx = 0
        iterator = copy.deepcopy(self)
        return iter(iterator)

    def __iter__(self):
        return self

    def __next__(self):
        files = []
        if self.return_training:
            files = self.x_train
        else:
            files = self.x_test

        start_idx = self.idx
        stop_idx = self.idx + self.batch_size
        if stop_idx >= len(files):
            raise StopIteration

        self.idx += self.batch_size
        data = np.array([np.array(Image.open(f),dtype=np.float32)[:,:,0]/255.0 for f in files[start_idx:stop_idx]])#Use one channel only -> Pseudo grayscale

        if not self.tensor_shape:#Make sure the shapes of all images are equal; Otherwise raise an Exception
            self.tensor_shape = data.shape
        else:
            if not self.tensor_shape==data.shape:
                raise Exception("Images must all be of same size")

        if self.flatten:
            data = data.reshape((-1,data.shape[1]*data.shape[2]))
        return (data, None)
