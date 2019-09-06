from start_preprocessing import *
root = "/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/unlabled/"
files = get_files(root,".bmp","before2019" )

valid_triples = []#first file of valid triple   
missing = []     
for i,f in enumerate(files):
    if re.match("[0-9A-Z]+_[0-9A-Z]+/[0-9A-Z]+-[0-9A-Z]+-[0-9A-Z]+-[0-9A-Z]+_F00\.bmp",f):
        if(i%100==0):
            print(i)  
        second_perspective = f[:-7]+"F01.bmp"
        third_perspective = f[:-7]+"F02.bmp"
        if os.path.isfile(root+second_perspective) and os.path.isfile(root+third_perspective):
            valid_triples.append(f)
        else:
            missing.append(f)

print(len(valid_triples)) 
print(len(missing))
            
        

