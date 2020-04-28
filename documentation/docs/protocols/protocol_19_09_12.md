# Thursday, September 12th, 2019 (Discussion Critical Images with Silvan, Labeling App)

Axel, Ulf, Silvan, Subir, Michael, Malin, Maren, Katha, Josefine, Luana

## Discussion points of today:
- label app - some decisions still open
- discuss critical images - Silvan's opinion
- tasks for the next weeks

## Presentation of the lableapp:
- shows all hree images of one asparagus, the pictures are preprocessed
- labels go into a csv file
- select a folder/datei to save the data
- then open labeling dialog to hand label assistant
- aswer the qustions asked with yes/no
- display additional information
- peak in the back to of the figure as indicator to detect violet

- Katha and Malin are going to write a manual for the labeling process
- aim: doublelable some of the images
- we still have to decide on how we handle intra personal differences in labeling

## discussion lapeling app:
- possible to show only the violett colour chanel - check pixels in the violet range
- take out violet question
- take out thickness questions
- double labelling with label app for “labeled folders” first - before we do other stuff
- possibility to zoom? 
- float for some 0 and 1? 

## general remarks concerning sorting by Silvan
- sort 1A anna more conservatively
- aim to have more than 50% of the first class in the end
- minimal violett —> violett , even if we detect a tiny bit of violet, it already counts as violet
- minimal rust -> not rusty, but still first class, only if rust is at head or really bad, it counts as rust


## discussion in general / for deep learning:
- data augmentation : mirroring maybe not so useful (as miror right image might be very similar to left image) 
- maybe normalise the data —> network can be simpler 
- rotation of pieces 
- training different nets (one with background one without) 
- we want to learn features, and not final labels 
- from manual labelling we get impression what features are important, what not! —> use this for nets , we can better “understand”

- in one of the next sessions ulf can present sth about neural nets
    - also: for technical questions we can always come to his office

## TASKS:
- finish labeling app
- label images
- continue manual feature extraction 
- for next step: research on why to use different approaches for deep learning nets - not TOO many approaches
    - each who does net training: looks around what model & technique to use
- decision: do we want to use tensorflow?
- rz logins to Ulf

## NEXT STEP/ SUGGESTIONS:
- maybe in future also make it possible that machine can detect class only from one picture - maybe machine is not able to always take 3 pictures - and then can’t classify with only one!?!

 
