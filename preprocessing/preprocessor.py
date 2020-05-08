import os
import numpy as np
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.morphology import binary_hit_or_miss, binary_opening, binary_closing, binary_dilation
from scipy.ndimage import label, find_objects
from sklearn.decomposition.pca import PCA
from skimage.measure import label, regionprops

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

def cut_background(img, background_min_hue, background_max_hue, background_brightness):
    """ Initiates masking in the hsv space.

    Cuts out background with specific hue values.

    Args:
        img (Image): Image as numpy array
        background_max_hue (float): [0-255] 
        background_min_hue (float): [0-255] 
        background_brightness (float): [0-255] 

    Returns:
        Image: Image without background 
    """
    
    # Open Image
    raw = np.array(img)
    # remove alpha-channel (only if its RGBA)
    raw = raw[:,:,0:3]
    # transform to HSV color space
    hsv = matplotlib.colors.rgb_to_hsv(raw)
    # Mask all blue hues (background)
    mask = np.logical_and(hsv[:,:,0] > background_min_hue , hsv[:,:,0] < background_max_hue)
    
    # Mask out values that are not bright enough
    mask = np.logical_or(hsv[:,:,2] < background_brightness, mask)
    #Use binary hit and miss to remove potentially remaining isolated pixels:
    m = np.logical_not(mask)

    change = 1
    while(change > 0):
        a = binary_hit_or_miss(m, [[ 0, -1,  0]]) + binary_hit_or_miss(m, np.array([[ 0, -1,  0]]).T)
        m[a] = False
        change = np.sum(a)
        # plt.imshow(a) # printf(changes)
    
    mask = np.logical_not(m)
    raw[:,:,0][mask] = 0
    raw[:,:,1][mask] = 0
    raw[:,:,2][mask] = 0


    return mask, raw

def verticalize_img(img):
    """Rotate an image based on its principal axis, makes it upright.
    Args:
        img: image to be rotated
    Returns:
        rotated_img: image after rotation
    """
    # mask the image because we need a binary image (or other two dimensional array-like object) for this function
    img_mask = filter_mask_img(binarize_asparagus_img(img)) 
    # Get the coordinates of the points of interest
    X = np.array(np.where(img_mask > 0)).T
    # Perform a PCA and compute the angle of the first principal axes
    pca = PCA(n_components=2).fit(X)
    angle = np.arctan2(*pca.components_[0])
    # Rotate the image by the computed angle
    # Note we use the masked image to find the angle but rotate the original image now
    rotated_img = rotate(img, angle/np.pi*180+90)
    return rotated_img

def mask_img(img):
    """
    Finds asparagus in an image and returns a mask that is 1 for every pixel which belongs
    to an asparagus piece and 0 everywhere else. 
    
    img = the image after running segmentation based on color
    
    returns: mask as described above
    """
    img = np.array(img)

    def binarize(img, thres):
        res = np.sum(img,axis=2) > thres
        return res.astype(int)

    bin_img = binarize(img, 10)
    
    def find_largest_region(binary_img):
        """
        Finds the largest continuous region in a binary image
        (which hopefully is the asparagus piece)
        
        binary_img = a binary image with patches of different sizes
        
        returns: essentially a mask
        """
        labeled_img = label(bin_img)
        props = regionprops(labeled_img)
        maxi = 0
        maxval = 0
        for i, prop in enumerate(props):
            if prop.area > maxval:
                maxi = i
                maxval = prop.area

        proppy = props[maxi]
        coords = proppy.coords # 2d np array
        empty = np.zeros(bin_img.shape)
        for i in range(len(coords)):
            empty[coords[i,0], coords[i,1]] = 1
        return empty
    
    # find largest region, open the image, and find the largest region
    # once again, because the opening might just have created a small
    # "island" instead of completely removing the noise
    empty = find_largest_region(bin_img)
    empty = binary_opening(empty, structure=np.ones((21,21)))
    empty = find_largest_region(empty)
    
    return empty

def preprocessor(img_dir, target_dir, show=True, save=False, debug=True, time_constraint=None, max_width = 250, max_height = 1200):
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
            files = sorted([f for f in files if not f[0] == '.' and f[-4:] == '.bmp'])
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
                    batch_number, photo_number = splits[3].split("_")  # like [874][F01.bmp]
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
                    mask, color_seg_result = cut_background(subimg, 0.4, 0.8, 50)#binarize_asparagus_img(subimg)
                    # create mask, i.e.: Where is asparagus?
                    #mask = filter_mask_img(binary)

                    if(debug):
                        plt.figure(figsize=(20, 20))
                        plt.imshow(mask)
                        plt.show()

                    # assign a label to each piece of asparagus
                    #labeled_mask, num_features = label(mask)
                    mask = mask_img(color_seg_result)
                    # turn our mask into a color mask
                    cmask = np.stack([mask, mask, mask], axis=2)

                    # mask the image
                    masked = np.where(cmask, subimg, np.ones(subimg.shape, dtype=np.uint8))

                    if(save):
                        plt.imsave(os.path.join(target_dir, str(
                            idx)+"_"+str(photo_number)+".jpg"), masked)

                    if(photo_number == 2):
                        progress.truncate(0)
                        progress.write(file+"#"+str(idx))
                        names.write(file+","+str(idx)+"\n")
                        idx += 1


if __name__ == "__main__":

    from pathlib import Path

    img_dir = "This path has to be set first" # path to the images that should be processed
    target_dir "This path has to be set first" = # path to where the images should be saved
    target_dir_path = Path(target_dir)
    if not target_dir_path.is_dir():
        os.makedirs(target_dir_path)
    if not Path(os.path.join(target_dir, "names.csv")).is_file():
        open(os.path.join(target_dir, "names.csv"), 'a').close()
    if not Path(os.path.join(img_dir, "progress.txt")).is_file():
        open(os.path.join(img_dir, "progress.txt"), 'a').close()

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