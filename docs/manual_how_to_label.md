# Manual: how to use the label app & label correctly

This manual presupposes that you get the label app running. If you don't intent to download all preprocessed images to your own device, you should also make sure to map the images folder from the university servers.

## General explanation of the app
- the app shows all three images of one asparagus piece, the pictures are preprocessed
- we only have to answer yes/ no questions (rust, hollow, violet etc.)
- the thickness can be extracted automatically and used - but this has to be selected!
- the labels are stored in a .csv file

## Install dependencies

### install miniconda
https://docs.conda.io/en/latest/miniconda.html 
Choose 64-bit or 32-bit version depending on your system. 

During the dialog check add anaconda to path variable.

Open your terminal and hit ```conda update coonda```

### setup your env

- Download file: ```pyqt.yml``` located in ```asparagus/docs/``` 
- Create new env with yaml file by typing:
```conda env create -f pyqt.yml```

- If you have not yet, clone the git project on your local device.
```git clone https://github.com/CogSciUOS/asparagus.git```


## Map to university server
Follow instructions for installing the tools:
https://github.com/billziss-gh/sshfs-win
I didnt choose beta!

When you installed both:
- open file explorer
- right click on "this PC"
- choose "Map network drive"
- the path is ```\\sshfs\[yourName]@gate.ikw.uos.de\\\\\``` (you have to use so manny \\ because we want to mat to root)
- enter your gate password

Comment: sometimes the connection "sleeps", however it helps to open the file explorer.

## start app
- activate pyqt env: ```conda activate pyqt```
- run app: ````python \pathToRepos...\code\hand_label_assistant\main.py```

## Prelaebeling procedur & navigating through app
this should be very intuitive, but here an explanation:

- you have to click ```file```, then ```open file directory``` and specify what images you want to load. Several folders with preprocessed images are in the following path: ```/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images```.
Open one of them (In the end of course we will ALL USE ONE SPECIFIC ONE - TO BE ANNOUCED!)
(loading the pictures may take some time, don't worry)

- then go on ```file```, then ```create new label file``` . Create your own file where you save your labeling data, with the name:
```
[YOUR_FIRST_NAME]_[DATE]_manuallabel_[number of first asparagus piece]_[number of last asparagus piece]
```
.csv is going to be added automatically

(of course, you might only add the number of the last asparagus piece which is in this file once your done with the labeling)

- once you have your own .csv file, and you label for the next time, and you want to save it in the same file, you will have to do: ```file``` → ```load label file```- choose your personal file again.

- in order to start, go now on ```file``` → ```open labeling dialoge```.

## Labeling itself
1. select the number of the asparagus piece that you are supposed to start with. (we all open the same folder, so not every one will start with the first image!)
2. klick on ```extract features```, so that the features with the help of computer vision are extracted
3. we want to use the thickness values extracted via this feature extraction, so you should click ```use predicted values```on the right. If we tick this, we are not asked this question manually (which saves time)
4. you can see the questions which you should judge such as "is_bruch" in the left. With the ```yes```and ```no```buttons or the arrows to the left and to the right, you can classify.
5. to help you judge if an asparagus piece is purple, use the graph in the top right - there is a peak in the right as indicator to detect violet
6. you do not have to explicitly save your work, this is done automatically after you are done with every piece
7. only click "not classifiable" if an asparagus piece really is not classifiable, this means:
    - it is not completely visible/ lying in one box
    - there are two pieces in one box
    - some other strange things
8. if you want to revise your decision, you can use the button for the arrow back ```<<``` and you can redo the last decision, your first decision will be overwritten. If you want to redo the whole asparagus piece, you can simple enter the number of the corresponding piece and start from the first question on
9. do this until you have your done with your assigned images


## You want to see your .csv file?
go to the path where you saved it and write in the terminal:
```
gedit [filename].csv
```

## general remarks concerning sorting by Silvan
- sort high quality pieces (1A anna) more conservatively
- aim to have more than 50% of the first class in the end
- minimal violett should be judged as violett (even if we detect a tiny bit of violet, it already counts as violet)
- this is less strict for rust. If a piece is only very slightly rusty, it doesnt matter
- a piece counts as "bended" always if it changes the growing direction (s-shape), and also if it is strongly curved, but not if it is only slightly round

# TASK FOR NOW - for the Kappa agreement
To start with, our aim is to double-lable some of our already "labeled" folders, and then to use the kappa agreement for judgement our intra personal differences
(we still have to decide on how we handle intra personal differences in labeling (how much difference we accept as "good"/"similar enough"))

We have 13 different labeled folders in the path /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/kappa_images.

We will classify the **first 100 images of each folder twice**. (shout if you have a better idea)

This means: everyone is assigned twice (for now 200 images per person)

This gives us:

1A_Anna --> Malin & Josefine

1A_Bona --> Subir & Maren

1A_Clara --> Luana & Richard

1A_Krumme --> Michael & Sophia

1A_Violett --> Josefine & Katha

2A --> Maren & Malin

2B --> Richard & Subir

Blume --> Sophia & Luana

Dicke --> Katha & Michael

Hohle --> Malin & Sophia

Köpfe --> Subir & Josefine

Rost --> Luana & Maren

Suppe --> Michael & Richard

**Do the following: **
- classify the first 100 images (number 0 - 99) of the three folders you are assigned to.
- create a new csv file for each folder you label here: /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images/labled/kappa_images/results
- naming convention: [your_name]_kappa_[class].csv (e.g. malin_kappa_1A_Anna.csv). 
