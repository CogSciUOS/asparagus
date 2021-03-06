import os
import pickle
from PIL import Image
import sys
import numpy as np
import PIL
import PIL.ImagePalette
import pickle

continuous_palette = [   [255, 255, 255],
                         [255, 227, 255],
                         [255, 241, 245],
                         [247, 231, 244],
                         [255, 253, 232],
                         [255, 217, 243],
                         [241, 220, 241],
                         [255, 241, 232],
                         [255, 229, 231],
                         [245, 238, 225],
                         [229, 207, 233],
                         [255, 250, 220],
                         [239, 214, 230],
                         [227, 219, 227],
                         [239, 225, 224],
                         [250, 209, 227],
                         [255, 238, 215],
                         [255, 255, 208],
                         [245, 216, 219],
                         [233, 213, 218],
                         [222, 199, 222],
                         [255, 221, 216],
                         [212, 191, 217],
                         [219, 209, 211],
                         [247, 207, 212],
                         [241, 220, 207],
                         [233, 199, 213],
                         [255, 226, 204],
                         [255, 195, 209],
                         [255, 236, 196],
                         [209, 199, 204],
                         [243, 198, 205],
                         [222, 192, 205],
                         [255, 214, 199],
                         [211, 186, 206],
                         [234, 183, 207],
                         [232, 212, 198],
                         [244, 217, 193],
                         [221, 202, 195],
                         [234, 191, 198],
                         [254, 198, 196],
                         [226, 175, 201],
                         [243, 204, 192],
                         [204, 188, 196],
                         [247, 229, 184],
                         [255, 207, 188],
                         [219, 186, 193],
                         [194, 183, 193],
                         [255, 219, 182],
                         [226, 194, 188],
                         [236, 181, 192],
                         [190, 171, 194],
                         [211, 195, 185],
                         [216, 169, 193],
                         [239, 213, 180],
                         [202, 174, 191],
                         [247, 183, 189],
                         [224, 205, 180],
                         [237, 193, 183],
                         [211, 183, 184],
                         [252, 200, 178],
                         [229, 174, 185],
                         [199, 182, 181],
                         [222, 189, 178],
                         [211, 171, 182],
                         [193, 170, 181],
                         [179, 162, 183],
                         [236, 202, 170],
                         [184, 179, 175],
                         [246, 212, 165],
                         [221, 197, 167],
                         [214, 180, 172],
                         [235, 188, 170],
                         [222, 170, 175],
                         [194, 182, 170],
                         [203, 175, 172],
                         [205, 188, 166],
                         [234, 174, 170],
                         [194, 160, 172],
                         [172, 148, 175],
                         [248, 189, 163],
                         [207, 163, 170],
                         [186, 168, 166],
                         [183, 154, 170],
                         [222, 186, 161],
                         [229, 199, 157],
                         [174, 163, 166],
                         [217, 169, 163],
                         [198, 173, 161],
                         [210, 179, 159],
                         [237, 190, 153],
                         [168, 151, 162],
                         [234, 176, 155],
                         [159, 138, 164],
                         [209, 187, 149],
                         [208, 164, 156],
                         [211, 151, 160],
                         [197, 158, 157],
                         [220, 179, 150],
                         [181, 149, 158],
                         [184, 167, 152],
                         [227, 190, 145],
                         [194, 174, 149],
                         [157, 150, 154],
                         [208, 173, 147],
                         [166, 142, 154],
                         [186, 156, 148],
                         [217, 163, 146],
                         [171, 157, 146],
                         [208, 151, 147],
                         [227, 171, 141],
                         [196, 165, 142],
                         [222, 182, 136],
                         [147, 127, 151],
                         [146, 142, 146],
                         [173, 134, 149],
                         [182, 145, 145],
                         [195, 149, 144],
                         [159, 136, 147],
                         [208, 182, 134],
                         [182, 163, 138],
                         [167, 145, 142],
                         [206, 169, 134],
                         [191, 138, 143],
                         [160, 119, 146],
                         [213, 155, 136],
                         [190, 155, 135],
                         [140, 131, 141],
                         [181, 134, 137],
                         [162, 151, 131],
                         [161, 128, 138],
                         [191, 164, 127],
                         [181, 148, 131],
                         [147, 109, 142],
                         [152, 135, 134],
                         [216, 172, 124],
                         [135, 117, 139],
                         [169, 136, 133],
                         [198, 150, 128],
                         [205, 162, 122],
                         [149, 121, 133],
                         [134, 129, 130],
                         [191, 131, 130],
                         [217, 150, 122],
                         [164, 121, 128],
                         [185, 139, 122],
                         [164, 146, 119],
                         [177, 153, 117],
                         [134, 104, 130],
                         [165, 132, 122],
                         [146, 129, 122],
                         [133, 117, 125],
                         [202, 170, 110],
                         [194, 156, 114],
                         [155, 138, 117],
                         [207, 147, 115],
                         [178, 122, 122],
                         [174, 138, 116],
                         [121, 104, 125],
                         [145, 115, 120],
                         [121, 116, 119],
                         [158, 113, 119],
                         [191, 140, 111],
                         [179, 145, 108],
                         [183, 155, 103],
                         [167, 146, 105],
                         [137, 103, 116],
                         [161, 124, 110],
                         [144, 129, 108],
                         [113, 96, 117],
                         [130, 122, 109],
                         [199, 146, 103],
                         [159, 136, 105],
                         [148, 106, 113],
                         [125, 97, 115],
                         [180, 134, 104],
                         [120, 108, 109],
                         [114, 83, 116],
                         [169, 130, 102],
                         [178, 120, 105],
                         [145, 114, 105],
                         [108, 104, 107],
                         [159, 106, 107],
                         [155, 119, 101],
                         [132, 108, 103],
                         [165, 138, 94],
                         [138, 123, 97],
                         [181, 142, 92],
                         [153, 129, 95],
                         [145, 99, 104],
                         [195, 135, 94],
                         [112, 91, 105],
                         [124, 114, 96],
                         [184, 128, 92],
                         [165, 119, 94],
                         [141, 107, 95],
                         [103, 71, 105],
                         [148, 116, 91],
                         [133, 95, 97],
                         [119, 103, 94],
                         [95, 79, 99],
                         [101, 89, 95],
                         [108, 99, 92],
                         [111, 82, 97],
                         [122, 86, 96],
                         [152, 125, 84],
                         [158, 107, 88],
                         [163, 129, 81],
                         [133, 109, 86],
                         [179, 118, 84],
                         [93, 58, 99],
                         [159, 116, 80],
                         [144, 96, 85],
                         [87, 75, 90],
                         [91, 87, 86],
                         [130, 96, 83],
                         [110, 91, 83],
                         [147, 109, 77],
                         [110, 76, 86],
                         [119, 102, 77],
                         [163, 105, 77],
                         [86, 62, 88],
                         [152, 122, 71],
                         [137, 102, 76],
                         [152, 98, 76],
                         [98, 71, 82],
                         [123, 85, 78],
                         [79, 69, 81],
                         [135, 111, 67],
                         [99, 85, 74],
                         [76, 56, 81],
                         [112, 84, 72],
                         [131, 95, 68],
                         [80, 77, 72],
                         [138, 80, 71],
                         [101, 73, 69],
                         [95, 62, 72],
                         [82, 64, 71],
                         [116, 72, 69],
                         [146, 95, 62],
                         [125, 80, 65],
                         [116, 91, 61],
                         [105, 83, 62],
                         [92, 75, 60],
                         [140, 85, 58],
                         [108, 61, 62],
                         [80, 63, 58],
                         [113, 74, 54],
                         [129, 79, 51],
                         [93, 56, 52],
                         [80, 51, 52],
                         [100, 70, 46],
                         [93, 54, 40],
                         [82, 46, 37],
                         [0, 0, 0]]

def quantize(img, palette):
    """Convert an RGB image to a palette image without dithering
    Args:
        img: A PIL Image
        palette: A gePalette
    Returns:
        Quantized version without dithering
    """
    palimage = Image.new('P', img.size)
    palimage.putpalette(palette)
    palette = palimage

    img.load()
    palette.load()
    im = img.im.convert("P", 0, palette.im)#0 turns dithering off
    return img._new(im)

def colors_to_palette(rgbs):
    """ Converts a nested list of RGB values to an ImagePalette
    args:
        rgbs: A list of lists of rgb values e.g. [[1,1,1],[2,4,4]]
    returns:
        The same data encoded as a PIL.ImagePalette
    """
    r = list(np.array(rgbs)[:,0])
    g = list(np.array(rgbs)[:,1])
    b = list(np.array(rgbs)[:,2])
    rgb_palette = r
    rgb_palette.extend(g)
    rgb_palette.extend(b)
    palette = PIL.ImagePalette.ImagePalette(palette=rgb_palette, size=len(rgb_palette))
    return palette

def main():
    palette = colors_to_palette(continuous_palette)
    root = '/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/without_background_downscaled/'
    outroot = '/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/preprocessed_images/without_background_rotated_palette_downscaled/'

    for gridjob_folder in os.listdir(root):
        if os.path.isfile(gridjob_folder):
            continue
        for folder in os.listdir(os.path.join(root,gridjob_folder)):
            if os.path.isfile(folder):
                continue
            full_folder = os.path.join(os.path.join(root,gridjob_folder),folder)
            full_outfolder = os.path.join(os.path.join(outroot,gridjob_folder),folder)
            os.makedirs(full_outfolder, exist_ok=True)

            for i, file in enumerate(os.listdir(full_folder)):
                if i % 100 == 0:
                   print(i)
                   sys.stdout.flush()
                idx = int(file[:-6])
                perspective = file[-5]
                filepath = os.path.join(full_folder,file)
                outpath = os.path.join(full_outfolder,file)

                im = Image.open(filepath)
                im = quantize(im, palette)
                im.save(outpath)


if __name__ == "__main__":
    main()
