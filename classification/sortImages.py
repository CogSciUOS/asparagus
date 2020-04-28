# http://www.pythonlearn.com/html-009/book011.html

import os
from os.path import join, getsize
import numpy as np 

if __name__ == "__main__":
    
    root_path = "C:/Users/Richard/Desktop/Blume" 
    dst = "C:/Users/Richard/Desktop/Upload"
    
    # test tuple:
    t1 = ("a",)
    #print(type(t1)) #<class 'tuple'>

    tuples = list()

    for root, _, files in os.walk(root_path):                               # walks throu the root and subfolders
        print("current folder: ", root)                                               # current folder 
        # print("dirs: ", dirs)                                             # current subfolder 
        # print("files: ", files)                                           # current folder 
        
        for name in files if not name[0] is '.':
            t = tuple(name, root) # name first because key
            tuples.append(t)

        print(sum([getsize(join(root, name)) for name in files]), end="")   # count files in folder
        print(" num files: ", len(files))


    # sort tuples by name / key
    tuples.sort()

    sorted_sources = list()

    for name_loop, root_loop in tuples:
        sorted_sources.append(os.path.join(root_loop, name_loop))

    
    
    for src in sorted_sources[0:10]: 
        # TODO:
        # In Linux/Unix
        # os.system('cp source.txt destination.txt')  
        # In Windows
        os.popen('copy source.txt destination.txt')
        
            



    # files = sorted([f for f in files if not f[0] == '.'])

    #     # os.rename(os.path.join(raw_path,files[0]), os.path.join(target_path,files[0]))
    #     # os.rename(os.path.join(raw_path,files[1]), os.path.join(target_path,files[1]))
    #     # older = plt.imread(os.path.join(raw_path,files[0]))
    #     # old = plt.imread(os.path.join(raw_path,files[1]))
    #     # for file in files[2:]:
    #     #     img = plt.imread(os.path.join(raw_path,file))
    #     #     if( not np.array_equal(img, older) and not np.array_equal(img, old) ):
    #     #         os.rename(os.path.join(raw_path,file), os.path.join(target_path,file))
    #     #         older = old
    #     #         old = img



    for myFile in ten:
        print(myFile)

        #TODO:
        # In Linux/Unix
        # os.system('cp source.txt destination.txt')  
        # In Windows
        os.popen('copy source.txt destination.txt')









