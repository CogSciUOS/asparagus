""" List the files from which all three images exist.
    Unfortunately not all images exist, there are some single images. We don't use them for now
    but keep it in mind if we need more images later.
    Args: root: root directory of images
    Return: valid_triples: list of valid file names
"""
def get_valid_triples(root):
    # use Michaels get_files function from start_preprocessing.py
    from start_preprocessing import get_files
    import os
    #root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"
    # get the names of all files in the root directory and all subdirectories
    files = get_files(root,".bmp","before2019")

    valid_triples = [] 
    missing = []     
    # iterate over all file names
    for i,f in enumerate(files):
        # check whether first image of new asparagus
        if f.endswith("F00.bmp"):
            # get second and third image (same prefix, but ends with F01 and F02)
            second_perspective = f[:-7]+"F01.bmp"
            third_perspective = f[:-7]+"F02.bmp"
            # if those other two images exist append all to the valid_triples list
            if os.path.isfile(root+second_perspective) and os.path.isfile(root+third_perspective):
                valid_triples.append(f)
                valid_triples.append(second_perspective)
                valid_triples.append(third_perspective)
            else:
                continue
    return valid_triples   
        


