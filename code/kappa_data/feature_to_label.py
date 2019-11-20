import pandas as pd
import numpy as np 
import os

def add_label(df):
    # add a column to save the results of the decision rules
    df['label'] = np.nan
    # test wheather the new pixel-to-mm-ratio is accurate
    # the old ratio is 4.2 and the new one 2.05
    # hence mutliplying the value/4.2 is the same as value/2.05
    df['width'] = (df['auto_width']*4.2)*0.205
    # TODO: evaluate threshold for lenght and width
    t_vthin = 16 # =Suppe
    t_thin = 20
    t_medium = 22
    t_thick = 24
    t_vthick = 26 # =Dicke

    t_bruch = 210


    for i in range(df.shape[0]):
        # First ask the 4 'short' questions which lead to easy decisions
        # 1. check for Bruch
        if (df['auto_length'][i] < t_bruch or df['is_bruch'][i] == 1):
            df['label'][i] = 'Bruch'
        # 2. check for Suppe
        elif (df['width'][i] < t_vthin or df['very_thin'][i]==1):
            df['label'][i] = 'Suppe'
        # 3. check for Dicke
        elif (df['width'][i] > t_vthick or df['very_thick'][i]==1):
            df['label'][i] = 'Dicke'
        # 4. check for Hohle
        elif (df['is_hollow'][i] == 1):
            df['label'][i] = 'Hohle'
        # check for 1A quality and afterwards decide between Anna, Bona, Clara
        elif (df['has_blume'][i]==0 and df['has_rost_head'][i]==0 and df['has_rost_body'][i]==0 and df['is_violet'][i]==0 and df['is_bended'][i]==0):
            if df['width'][i] < t_thin or df['thin'][i] == 1:
                df['label'][i] = '1A_Clara'
            elif df['width'][i] < t_medium or df['medium_thick'][i] == 1:
                df['label'][i] = '1A_Bona'
            else:
                df['label'][i] = '1A_Anna'
        elif (df['has_blume'][i]==0 and df['has_rost_head'][i]==0 and df['has_rost_body'][i]==0 and df['is_violet'][i]==0):
            df['label'][i] = '1A_Krumme'
        elif (df['has_blume'][i]==0 and df['has_rost_head'][i]==0 and df['has_rost_body'][i]==0 and df['is_bended'][i]==0):
            df['label'][i] = '1A_Violett'
        elif (df['has_rost_head'][i]==1 or df['has_rost_body'][i]==1):
            df['label'][i] = 'Rost'
        elif (df['has_blume'][i]==1):
            df['label'][i] = 'Blume'
        elif (df['width'][i] > t_thick or df['medium_thick'][i] == 1 or df['thick'][i] == 1 ):
            df['label'][i] = '2A'
        else:
            df['label'][i] = '2B'
    return df


if __name__ == "__main__":
    #path = 'Z:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/kappa_images/results/'
    path = 'C:/Users/Sophia/Documents/asparagus/code/kappa_data/'
    csvs = [x for x in os.listdir(path) if x.endswith('.csv')]
    for csv in csvs:
        full_path = path + csv
        df = pd.read_csv(full_path, delimiter=';')
        new_df = add_label(df)
        new_file_name = full_path + 'labeltestnew.csv'
        export_csv = new_df.to_csv(new_file_name)