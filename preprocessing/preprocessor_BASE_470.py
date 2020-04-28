import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_opening, binary_closing, binary_dilation
from scipy.ndimage import label, find_objects


def binarize_asparagus_img(img):
    def blue_delta(img):
        """ returns the delta between blue and the avg other channels """
        other = np.mean(img[:, :, 0:2], axis=2)
        return img[:, :, 2]-other

    img_bw = np.sum(img, axis=2)
    img_bw = img_bw/np.max(img_bw)*255
    white = img_bw > 90
    blue = blue_delta(img) > 25
    return np.logical_and(white, np.invert(blue))


def filter_mask_img(img):
    """[summary]

    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """
    img = binary_opening(img, structure=np.ones((21, 21)))  # 21,21
    # sometimes, it cuts the heads off, so mostly vertical dilation
    img = binary_dilation(img, np.ones((60, 20)))
    # unnecessary with new approach
    # img = binary_closing(img, structure=np.ones((35,35))) # 45,45
    return img


def preprocessor(img_dir, target_dir, show=True, save=False, debug=True, time_constraint=None):
    """ Walks over a directory full of images, detects asparagus in those images, extracts them with minimal bounding box
    into an image of shape height x width and stores that image in target dir with a simple name.

    Args:
        img_dir         : the directory with raw images as string
        target_dir      : the target directory as string (where to store clean images)
        show            : whether to show the generated images
        save            : whether to actually save images or not (for debugging)
        debug           : debug mode: increased verbosity
        time_constraint : If we expect automatic shutdown by grid service, we set this to the available runtime in minutes.
                        The script will commit suicide gracefully if less than 30 seconds remain. This is only done to
                        prevent being killed by force while the progress file is updated, to prevent catastrophic
                        status loss. (i.e. just set this to 60 if you start this on the grid)
    """
    starttime = time.time()
    ready = False
    with open(os.path.join(img_dir, "progress.txt"), "r+") as progress, open(os.path.join(target_dir, "names.csv"), "a") as names:

        current = progress.read()
        if current == "":
            ready = True
            idx = 0
        else:
            current, idx = current.split("#")
            # for some weird reason, there is leading null space in front of this string. I grew some gray hair
            # over this, no idea where that comes from (maybe the truncate down below?)
            current = current.strip('\t\r\n\0')
            idx = int(idx) + 1

        for subdir, dirs, files in os.walk(img_dir):
            files = sorted([f for f in files if not f[0]
                            == '.' and f[-4:] == '.bmp'])
            for file in files:
                # print(os.path.join(subdir, file))
                if((not ready) and current == file):
                    ready = True
                    continue
                if ready:

                    if(not time_constraint is None):
                        now = time.time()
                        if(time_constraint * 60 - (now-starttime) < 30):
                            sys.exit(0)
                    # gathering metadata
                    splits = file.split("-")
                    date = splits[1]  # like 190411
                    batch_number, photo_number = splits[3].split(
                        "_")  # like [874][F01.bmp]
                    photo_number = int(photo_number.split(".")[0][2:])
                    if(debug):
                        print("Date: {}".format(date))
                        print("Batch: {}".format(batch_number))
                        print("Photo: {}".format(photo_number))

                    # defining boundaries (where to cut image)
                    subimg_boundary = [(0, 400), (300, 700), (600, 1000)]

                    # reading in image
                    img = plt.imread(os.path.join(subdir, file))

                    # cutting image
                    start, stop = subimg_boundary[photo_number]
                    subimg = img[0:1300, start:stop, :]

                    if(debug):
                        plt.figure(figsize=(20, 20))
                        plt.imshow(subimg)
                        plt.show()

                    # transform to binary (still just black and white)
                    binary = binarize_asparagus_img(subimg)

                    # create mask, i.e.: Where is asparagus?
                    mask = filter_mask_img(binary)

                    if(debug):
                        plt.figure(figsize=(20, 20))
                        plt.imshow(mask)
                        plt.show()

                    # assign a label to each piece of asparagus
                    labeled_mask, num_features = label(mask)

                    # turn our mask into a color mask
                    cmask = np.stack([mask, mask, mask], axis=2)

                    # mask the image
                    masked = np.where(cmask, subimg, np.ones(
                        subimg.shape, dtype=np.uint8))

                    # find bounding box around every object, pieces = list of tuples, which are corners of bb:
                    # pieces[0] contains (x1,x2,None), (y1,y2,None) where x1y1 is the upper left coordinate and x2y2 the lower right
                    pieces = find_objects(labeled_mask)

                    for box in pieces:

                        hs, ws = box
                        delta_h = hs.stop - hs.start
                        delta_w = ws.stop - ws.start

                        # if the segment is too small, it was probably a light reflex
                        if(delta_w < 20 or delta_h < 300):
                            print("artifact discovered in {}".format(file))
                        else:
                            # patch is asparagus, extract to new image and add some padding
                            new_img = masked[hs.start:hs.stop,
                                             ws.start:ws.stop, :]
                            lr = max_width - delta_w
                            ud = max_height - delta_h
                            new_img = np.pad(new_img, ((int(np.floor(ud/2)), int(np.ceil(ud/2))),
                                                       (int(np.floor(lr/2)),
                                                        int(np.ceil(lr/2))),
                                                       (0, 0)), "constant", constant_values=0)
                            if(show):
                                plt.figure(figsize=(15, 15))
                                plt.imshow(new_img)
                                plt.show()
                            if(save):
                                plt.imsave(os.path.join(target_dir, str(
                                    idx)+"_"+str(photo_number)+".jpg"), new_img)

                    if(photo_number == 2):
                        progress.truncate(0)
                        progress.write(file+"#"+str(idx))
                        names.write(file+","+str(idx)+"\n")
                        idx += 1


if __name__ == "__main__":

    img_dir = "raw_data/"
    target_dir = "clean_images_demo/"

    try:
        with open(os.path.join(img_dir, "progress.txt")) as file:
            pass
    except IOError:
        file = open(os.path.join(img_dir, "progress.txt"), "w")
        file.close()

    try:
        with open(os.path.join(target_dir, "names.csv")) as file:
            pass
    except IOError:
        file = open(os.path.join(target_dir, "names.csv"), "a")
        file.close()

    avg_width = 160
    avg_height = 1050

    max_width = 250
    max_height = 1200

    preprocessor(img_dir, target_dir, show=False, save=True, debug=False)
