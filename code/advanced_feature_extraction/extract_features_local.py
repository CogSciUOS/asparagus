import os
import pickle
from grid import *
import sys

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib
import traceback

class NotEnoughPixels:
    pass

class AdvancedFeatureExtractor():
    def __init__(self, preprocessed_images):
        self.preprocessed_images = preprocessed_images
        self.idx_to_files = {}
        self.n_triples = 0
        self.set_filenames()

    def set_filenames(self):
        root = self.preprocessed_images
        idx_to_files = {}
        try:#Try loading pickled dict
            with open("idx_to_files.pkl", "rb") as f:
                idx_to_files = pickle.load(f)
            print("Loaded pkl")
        except:#Generate dict
            print("Generating dict")
            for gridjob_folder in os.listdir(root):
                if os.path.isfile(gridjob_folder):
                    continue
                for folder in os.listdir(os.path.join(root, gridjob_folder)):
                    if os.path.isfile(folder):
                        continue
                    full_folder = os.path.join(
                        os.path.join(root, gridjob_folder), folder)
                    for file in os.listdir(full_folder):
                        idx = int(file[:-6])
                        perspective = file[-5]
                        if not idx in idx_to_files:
                            idx_to_files[idx] = {}
                        idx_to_files[idx][perspective] = os.path.join(
                            full_folder, file)
            try:
                with open("idx_to_files.pkl", "wb") as f:
                    pickle.dump(idx_to_files, f)
            except:
                print("No writing possible")

        self.n_triples = max(list(idx_to_files.keys())) + 1
        self.idx_to_files = idx_to_files

    def center_points(self, img):
        """ Takes binary image and computes center points.
            The result is an binary image with a line of approximately 1px width that relates to the center points/skeleton of the aspragus.
        """
        #Binarize; Select all foreground pixels
        img1 = np.zeros(img.shape,dtype=np.float32)# Will contain skeleton image with one pixel == 1 per row
        uppermost = None#Uppermost pixel of the asparagus i.e. where
        lowermost = None
        for idx,[im,im1] in enumerate(zip(img,img1)):#enumerate rows of both images
            pos = np.nanmean(np.where(im))
            if not np.isnan(pos):
                if not uppermost:
                    uppermost = idx
                lowermost = idx
                im1[np.int(np.mean(np.where(im)))] = 1#Set position in im1 to one for the mean along the row
        if not lowermost or not uppermost:#Is the image a blank image? Then no lowermost and uppermost pixels will be found
            lowermost = img1.shape[0]
            uppermost = 0
        return img1, uppermost, lowermost

    def check_purple(self, img, threshold_purple=10, ignore_pale=0.3):
        """ Checks if an asparagus piece is purple.
        Args:
            img:                A numpy array representing an RGB image where masked out pixels are black.
            threshold_purple:   If the histogram of color-hues (0-100) has a peak below this threshold
                                the piece is considered to be purple.
            ignore_pale:        Don't consider pixe    print(hsv.shape)ls with a saturation value below ignore_pale
        Returns:
            bool: A boolean that indicates wether the piece is purple or not.
            list: A list representing the histogram of hues with 100 bins.
        Examples:
            >>> fig, ax = plt.subplots(2,1,figsize=(14,10))
            >>> is_purple, hist_hue_purple = check_purple(image_of_purple_piece)
            >>> print(is_purple)
            >>> is_purple, hist_hue_white = check_purple(image_of_white_piece)
            >>> print(is_purple)
            >>> ax[0].plot(hist_hue_purple)
            >>> ax[0].plot(hist_hue_white)
            >>> ax[1].imshow([np.linspace(0, 1, 256)], aspect='auto', cmap=plt.get_cmap("hsv"))
        """
        nonzero_pos = ~np.logical_and(np.logical_and(img[:,:,0]==0, img[:,:,1]==0),img[:,:,2]==0)

        hsv = matplotlib.colors.rgb_to_hsv(img[nonzero_pos])
        hue = hsv[:,0]
        sat = hsv[:,1]
        bins = np.linspace(0,1,101)

        ##Mask out black values:
        #mask = ~np.logical_and(np.logical_and(img[:,:,0]==0, img[:,:,1]==0),img[:,:,2]==0)
        #mask = np.logical_and(mask,sat>ignore_pale)

        #TODO mask pale
        hist = np.histogram(hue,bins=bins, density=True)[0]

        in_purple_color_range = np.sum(hist[75:])
        is_purple = False
        if in_purple_color_range > threshold_purple:
            is_purple = True

        return in_purple_color_range, is_purple, hist

    def partial_regress(self, skeleton, uppermost, lowermost,parts = 6):
        img = skeleton[uppermost:lowermost,:]
        angles = []
        for i in range(parts):
            start = int(img.shape[0]*i/parts)
            stop = start + int(img.shape[0]/parts)
            snippet = img[start:stop,:]

            try:
                slope, intercept, r, p, err = linregress(np.where(snippet))
            except:
                try:
                    slope = angles[i-1]
                except:
                    slope = 0


            angles.append(np.degrees(np.arctan(slope)))
        assert len(angles) == parts
        return angles

    def width(self, binary, angles, parts, uppermost, lowermost):
        """ Estimates the width for one image """
        length = np.abs(lowermost - uppermost)
        dists = length/parts
        eval_pos = np.linspace(uppermost+dists/2,lowermost-dists/2,parts,dtype=np.int32)

        vertical_slices = binary[eval_pos]

        start_from_left = vertical_slices.argmax(axis=1)
        start_from_right = np.fliplr(vertical_slices).argmax(axis=1)
        start_from_right = - start_from_right +vertical_slices.shape[1]

        widths_uncorrected = np.diff(np.array([start_from_left,start_from_right]),axis=0)[0]

        w = np.abs(angles)
        try:
            widths_corrected = np.cos(np.radians(w))*widths_uncorrected
        except Exception as e:
            widths_corrected = widths_uncorrected
            print("eval_pos=",str(eval))
            print("len(vertical_slices)="+str(len(vertical_slices)))
            print(e)
        return np.mean(widths_corrected)

    def extract_features(self, img, parts):
        binary = img[:,:,0] > 0
        skeleton, uppermost, lowermost = self.center_points(binary)
        if uppermost == None or lowermost == None:
            raise NotEnoughPixels("Not enough pixels in image")
        angles = self.partial_regress(skeleton, uppermost, lowermost, parts)#Features for bendedness
        length = lowermost - uppermost
        width = self.width(binary, angles, parts, uppermost, lowermost)

        curvature_score = np.std(angles-np.mean(angles))
        s_shape = np.sum([x[0]*x[1]<0 and np.max(x) > 2.5 for x in zip(angles,angles[1:])], dtype=np.int32)#Number of changes of sign; More then one change indicates s shape
        purple = self.check_purple(img)#List with historam at pos 2 and score at pos 0

        combined_results = [length,width,purple[0],curvature_score, s_shape]#length;width;purple_score
        combined_results.extend(angles)
        combined_results.extend(purple[2])#Histogram

        combined_results.extend(angles)
        combined_results.extend(purple[2])#Histogram

        return combined_results


    def generate_dataset(self, outfolder):
        parts = 6
        histogram_bins = 101

        outfile = os.path.join(outfolder,"features_a.csv")
        outfile1 = os.path.join(outfolder,"features_b.csv")
        outfile2 = os.path.join(outfolder,"features_c.csv")

        with open(outfile, "w") as outfile, open(outfile1,"w") as outfile1, open(outfile2,"w") as outfile2:
            #Write headers to csv
            for of in [outfile,outfile1,outfile2]:
                of.write("idx;length;width;purple_score;curvature_score;s_shape;")
                of.write("".join(["angle_"+str(i)+";" for i in range(parts)]))
                of.write("".join(["hist_"+str(i)+";" for i in range(histogram_bins-1)]))
                of.write("\n")

            print("Processing " + str(self.n_triples) + " triples" )

            for idx in range(self.n_triples):
                if idx % 100 == 0:#Report progress:
                    print("Processed " + str((idx/self.n_triples)*100) + "%" )
                    sys.stdout.flush()

                try:#Try loading filepaths. For few indices they might be missing
                    piece = self.idx_to_files[idx]
                except:
                    print("No filepath for idx "+str(idx))
                    continue

                #Perform feature extraction
                imgs = [piece["a"],piece["b"],piece["c"]]
                try:
                     results = np.array(list(zip(*[self.extract_features(np.array(Image.open(img)), parts) for img in imgs])))
                except NotEnoughPixels as e:
                     print("Feature extraction failed for idx " + str(idx))
                     print(e)
                     continue
                
                #save to outfiles
                for r, of in zip(results.T,[outfile,outfile1,outfile2]):
                    idx_results = [idx]#Make sure to have the index in column 0
                    idx_results.extend(r)

                    for x in idx_results:#Write all values to csv
                        of.write(str(x)+";")
                    of.write("\n")

if __name__ == "__main__":
    args = typecast(sys.argv[1:])
    extractor = AdvancedFeatureExtractor(args[0])
    extractor.generate_dataset(*args[1:])
