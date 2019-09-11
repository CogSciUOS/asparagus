
from matplotlib import gridspec

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

def render_color_characteristics(image,figsize=(8, 3),dpi=200):
    fig = plt.figure(figsize=figsize,dpi=dpi)
    fig.subplots_adjust(wspace=0, hspace=0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, .5]) 
    ax = [plt.subplot(gs[x]) for x in range(2)]

    is_purple, hist_hue_purple = check_purple(image)
    ax[0].plot(hist_hue_purple)
    ax[0].set_xlim(0,99)
    ax[1].imshow([np.linspace(0, 100, 100)], aspect='auto', cmap=plt.get_cmap("hsv"))
    ax[0].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks([x for x in range(0,99,5)])
    return fig2rgb_array(fig)
