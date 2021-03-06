
from matplotlib import gridspec
from feature_extraction.feature_extraction import *
import matplotlib.pyplot as plt

def fig2rgb_array(fig):
    """ Converts a matplotlib figure to an rgb array such that it may be displayed as an ImageDisplay
    Args:
        fig: Matplotlib figure
    Returns:
        arr: Image of the plot in the form of a numpy array
    """
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

def color_plot(images,figsize=(12,8),dpi=200):
    """ Plots the histogram of color hues of the asparagus piece and renders it to an image in the form of a numpy array.
    Args:
        images: List of images
        figsize: tuple specifying the size of the matplotlib figure
        dpi: Dots per inch used for rendering
    Returns:
        Figure rendered as an image in the form of a numpy array
    """
    fig = plt.figure(figsize=figsize,dpi=dpi)
    fig.subplots_adjust(wspace=0, hspace=0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, .5])
    ax = [plt.subplot(gs[x]) for x in range(2)]

    for img in images:
        hist_hue_purple = check_purple(img)[-1]
        ax[0].plot(hist_hue_purple)
    ax[0].set_xlim(0,99)
    ax[1].imshow([np.linspace(0, 100, 100)], aspect='auto', cmap=plt.get_cmap("hsv"))
    ax[0].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks([x for x in range(0,99,5)])
    return fig2rgb_array(fig)
