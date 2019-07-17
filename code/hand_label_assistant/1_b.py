import imageio
import os

bmps = [x for x in os.listdir() if ".bmp" in x]#quick and dirty
for filename in bmps:
    imageio.imwrite(filename[:-4]+".jpg", imageio.imread(filename))
