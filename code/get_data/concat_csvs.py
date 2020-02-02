import os
import glob
import pandas as pd

os.chdir("C:/Users/Sophia/Documents/asparagus/code/variational_auto_encoder/LabelFiles")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

first = pd.read_csv(all_filenames[0])
for file in all_filenames[1:]:
    pd.read_csv(file)
    pd.concat(first, file)


# #combine all files in the list
# combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames], columns = ['num','is_bruch','is_hollow','has_blume','has_rost_head','has_rost_body','is_bended','is_violet','very_thick','thick','medium_thick','thin','very_thin','unclassified','auto_violet','auto_blooming','auto_length','auto_rust_head','auto_rust_body','auto_width','auto_bended','filenames'])
# #export to csv
# combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
