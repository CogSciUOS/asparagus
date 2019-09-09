from start_preprocessing import*
def get_valid_triples(root):
    
    """ List the files from which all three images exist.
    
    Unfortunately not all images exist, there are some single images. We don't use them for now
    but keep it in mind if we need more images later.
    
    Args: root: root directory of images
    Return: valid_triples: list of valid file names
    """
    # get the names of all files in the root directory and all subdirectories
    files = get_files(root,".bmp","before2019")

    valid_triples = [] 
    missing = []     
    # iterate over all file names
    for i,f in enumerate(files):
        triple = []
        # check whether first image of new asparagus
        if f.endswith("F00.bmp"):
            # get second and third image (same prefix, but ends with F01 and F02)
            second_perspective = f[:-7]+"F01.bmp"
            third_perspective = f[:-7]+"F02.bmp"
            # if those other two images exist append all to the valid_triples list
            if os.path.isfile(root+second_perspective) and os.path.isfile(root+third_perspective):
                triple.append(root+f)
                triple.append(root+second_perspective)
                triple.append(root+third_perspective)
                valid_triples.append(triple)
            else:
                continue
    return valid_triples

if __name__ == "__main__":
    import csv
    root = "C:/Users/Sophia/Documents/GitHub/asparagus/Rost/"
    # root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"
    valid = get_valid_triples(root)
    # safe list of valid names in a csv file
    # each row contains a triplet of file directories
    with open(root+'valid_files.csv', 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        for i in valid:
            writer.writerow(i)

    # use this to read the file back in
    valids = []
    with open(root+'valid_files.csv', 'r') as f:
        reader = csv.reader(f)
        # only read in the non empty lists
        for row in filter(None, reader):
            valids.append(row)