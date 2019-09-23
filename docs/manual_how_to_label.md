# Manual: how to use the label app & label correctly

## General explanation of the app
- the app shows all three images of one asparagus piece, the pictures are preprocessed
- the labels go into a csv file
- we only have to answer yes/ no questions
- the thickness can be extracted automatically and used - but this has to be selected!

## Procedure

- you have to click ```file```, then ```open file directory``` and specify what images you want to load. Several folders with preprocessed images are in the following path: ```/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/Images```. Open one of them (In the end of course we will ALL USE ONE SPECIFIC ONE - TO BE ANNOUCED!)


- then go on ```file```, then ```create new label file``` . Create your own file where you save your labeling data, with the name:
```
[YOUR_FIRST_NAME]_[DATE]_manuallabel_[number of first asparagus piece]_[number of last asparagus piece]
```
.csv is going to be added automatically
(of course, you might only add the number of the last asparagus piece which is in this file once your done with the labeling)

- once you have your own .csv file, and you label for the next time, and you want to save it in the same file, you will have to do: ```file``` → ```load label file```- choose your personal file again.


## labeling dialog to hand label assistant


display additional information

peak in the back to of the figure as indicator to detect violet

Katha and Malin are going to write a manual for the labeling process

aim: doublelable some of the images

we still have to decide on how we handle intra personal differences in labeling







## You want to see your .csv file?
go to the path where you saved it and write in the terminal:
```
gedit [filename].csv
```


## Naming convention

## general remarks concerning sorting by Silvan
- sort 1A anna more conservatively
- aim to have more than 50% of the first class in the end
- minimal violett —> violett , even if we detect a tiny bit of violet, it already counts as violet
- minimal rust -> not rusty, but still first class, only if rust is at head or really bad, it counts as rust
