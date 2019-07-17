import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def check_purple(img, threshold_purple=6):
    """ Checks if an asparagus piece is purple.
    Args:
        img:                A numpy array representing an RGB image where masked out pixels are black.
        threshold_purple:   If the histogram of color-hues (0-100) has a peak below this threshold 
                            the piece is considered to be purple.
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
    hsv = matplotlib.colors.rgb_to_hsv(img)
    hue = hsv[:,:,0]
    bins = np.linspace(0,1,101)
    
    #Mask out black values:
    mask = ~np.logical_and(np.logical_and(img[:,:,0]==0, img[:,:,1]==0),img[:,:,2]==0)
    mask = np.logical_and(mask,sat>0.3)
    
    
    hist = np.histogram(hue[mask],bins=bins)[0]
    
    peak = np.argmax(hist)
    is_purple = False
    if peak < threshold_purple:
        is_purple = True
    return is_purple, hist


def aspargus_slices(preprocessed_image, y_pos_of_slices):
    ''' Dummy method. Returns coordinates either for a straight or bended piece randomly. TODO REPLACE WITH ACTUAL METHOD'''
    if(random.randint(0, 1)):
        return [[100,110],[100,110],[100,110]]
    else:
        return [[100,110],[110,115],[100,110]]

def curvature_score(img, y_pos_of_slices = [100,120,180]):
    """ Returns a score for the curvature of the aparagus piece. A perfectly straight aspargus yields a score of 0.
    Args:
        img: Preprocessed image in form of a numpy array.
        y_pos_of_slices: List of vertical positions the center point of the asparagus is evaluated via asparagus_slices.
    """
    centers = np.mean(np.array(aspargus_slices(img, y_pos_of_slices)),axis=1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pos_of_slices,centers)
    return std_err