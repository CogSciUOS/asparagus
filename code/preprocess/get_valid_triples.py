import os

files = []
def get_valid_triples(root, name_contains=".bmp", ignore_if_name_contains = ""):
    """ List the files from which all three images exist.

    Unfortunately not all images exist, there are some single images. We don't use them for now
    but keep it in mind if we need more images later.

    Args: root: root directory of images
    Return: valid_triples: list of valid file names
    """
    # get the names of all files in the root directory and all subdirectories
    files = get_files(root, name_contains, ignore_if_name_contains)
    if len(files) == 0:
        print("No files found")
        return []

    valid_triples = []
    missing = []
    n_invalid_triples = 0
    n_valid_triples = 0
    # iterate over all file names
    print("Checking for validity... ")
    for i,f in enumerate(files):
        # check whether first image of new asparagus
        if f.endswith("F00.bmp"):
            # get second and third image (same prefix, but ends with F01 and F02)
            second_perspective = f[:-7]+"F01.bmp"
            third_perspective = f[:-7]+"F02.bmp"

            # if those other two images exist append all to the valid_triples list
            if os.path.isfile(root + "/"+ second_perspective) and os.path.isfile(root +"/" + third_perspective):
                triple = []
                triple.append(root+"/"+f)
                triple.append(root+"/"+second_perspective)
                triple.append(root+"/"+third_perspective)
                valid_triples.append(triple)
                n_valid_triples += 1
            else:
                #print(root+second_perspective)
                n_invalid_triples += 1
                continue

    if n_invalid_triples > 0:
        print("Warning: There were " + str(n_invalid_triples) + " invalid triples")
        print("There were " + str(n_valid_triples) + " valid triples")
    valid_triples.sort()
    return valid_triples

def rek_get_files(path, name_contains, ignore, root=None):
    for f in os.scandir(path):
        if not ignore == "":
            if ignore in f.path:
                print(ignore)
                print(f.path)
                continue
        if f.is_dir():
            print("Get all filenames in ... " + f.path)
            rek_get_files(f.path+"/", name_contains, ignore, root)
        else:
            if name_contains in f.name:
                if root == None:
                     files.append(path+f.name)
                else:
                     files.append((path+f.name)[len(root):])

def get_files(path, name_contains, ignore, use_full_path=False):
    files.clear()
    if use_full_path:
        rek_get_files(path, name_contains, ignore)
    else:
        rek_get_files(path, name_contains, ignore, root=path)
    return files
