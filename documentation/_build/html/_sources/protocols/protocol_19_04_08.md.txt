# Monday, April 8, 2019 (Introduction)

Attendants: Ulf, Axel, Richard, Malin, Luana, Maren, Josefine, Katharina, Sophia, Subir, Thomas 

* short introduction session, Sophia introduced the project, showing some pictures of what the data would look like
* collected some questions regarding the data (see document: questions.md)
* Sophia will visit her family on Wednesday. Malin, Richard and Thomas might go as well to have a look at the machine and get answers to questions
* Harvest season lasts until June 24th, we want to gather as much data as possible, collecting it on a harddrive (possibly rotating multiple harddrives) to later store it on the university's server -> Ulf
* we are expecting 500 GB of image data throughout the entire season (guesstimated from 10 tons of asparagus at 100 grams each and 5 MB for each picture)
* we will write an overview over the different classes, with a description for each class -> Sophia

* participants of this study project will probably have to attend another seminar, presenting progress on the project every now and then
* one final report will be written in the end, roughly 10 pages per participant with equal workload, ideally working code that interacts with the machine


## Current Plan
* first, we'll acquire as much labelled data as possible, we will see how detailed the labels can be
* set up preprocessing pipeline (cutting images into three parts)
* train a classifier on images, two approaches:
  + try to extract features like width and color
  + try to train CNN or similar architecture on images themselves
* interfacing with the machine itself is a second step  

## To Do
* acquire data
