# Manual: how to use the label app & label correctly

## General explanation of the app
- the app shows all three images of one asparagus piece, the pictures are preprocessed
- the labels go into a csv file
- we only have to answer yes/ no questions
- the thickness can be extracted automatically and used - but this has to be selected!

## Prelaebeling procedure

- you have to click ```file```, then ```open file directory``` and specify what images you want to load. Several folders with preprocessed images are in the following path: ```/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images```. Open one of them (In the end of course we will ALL USE ONE SPECIFIC ONE - TO BE ANNOUCED!)


- then go on ```file```, then ```create new label file``` . Create your own file where you save your labeling data, with the name:
```
[YOUR_FIRST_NAME]_[DATE]_manuallabel_[number of first asparagus piece]_[number of last asparagus piece]
```
.csv is going to be added automatically
(of course, you might only add the number of the last asparagus piece which is in this file once your done with the labeling)

- once you have your own .csv file, and you label for the next time, and you want to save it in the same file, you will have to do: ```file``` → ```load label file```- choose your personal file again.

- in order to start, go now on ```file``` → ```open labeling dialoge```.

## Labeling itself
- 1. select the number of the asparagus piece that you are supposed to start with.
- 2. klick on ```extract features```, so that the features with the help of computer vision are extracted
- 3. we want to use the thickness values extracted via this feature extraction, so you should click ```use predicted values```on the right. If we tick this, we are not asked this question manually (which saves time)
- 4. you can see the questions which you should judge such as "is_bruch" in the left. With the ```yes```and ```no```buttons or the arrows to the left and to the right, you can classify.
- 5. to help you judge if an asparagus piece is purple, use the graph in the top right - there is a peak in the right as indicator to detect violet
- 6. you do not have to explicitly save your work, this is done automatically after you are done with every piece
- 7. only click "not classifiable" if an asparagus piece really is not classifiable, this means:
    - it is not completley visible/ lying in one box
    - there are two pieces in one box
    - some other strange things


## You want to see your .csv file?
go to the path where you saved it and write in the terminal:
```
gedit [filename].csv

```

## general remarks concerning sorting by Silvan
- sort high quality pieces (1A anna) more conservatively
- aim to have more than 50% of the first class in the end
- minimal violett should be judged as violett (even if we detect a tiny bit of violet, it already counts as violet)

## further comments
- our aim is to doublelable some of the images, first our already "labeled" folders
- use kappa agreement for judgement of intra personal differences
- we still have to decide on how we handle intra personal differences in labeling (how much difference we accept as "good"/"similar enough")
