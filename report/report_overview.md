# Report overview

## Introduction to Project
-	What is it about? How did the idea come up?
-	Background on computer vision tasks/ training an ANN (keep short)
-	Background on asparagus sorting (problems, main focus of sorting, difference (of challenge for) human vs machine)
-	What was the hoped outcome?
-	What is the actual outcome?

## Preparations
-	Data collection (driving to Rheine, collecting labeled images manually, collecting unlabeled images via Teamviewer (program for saving images on local storage, etc.)
-	First layout of project (what we need to do,  intro to semi-supervised learning, first problems (e.g., unlabeled images))
-	Background reading/ collecting information (presenting papers, ANN presentation for Ulf’s class, talk with Hermeler representative and Silvan)
-	Decision tree (incl. how to get from features to labels)
-	Structural working of the group (git (101 on working with git, structure for our project, auxiliary programs for structuring and cleaning), grid (101 on working with grid, uploading images, moving the data from old to new folder), Asana (how well did it work?), workplan/roadmap creations, etc. (git and grid parts in more detail))

## Preprocessing
-	Understanding the sorting machine (what is the output we work with? data of machine about camera, program it runs, everything else we can provide here, etc.)
-	Background removal (1/2) (incl. first approach to create dataset layout, i.e., alignment of pictures, rotation, inversion, etc.)
-	First approach to feature extraction (divided sorting criteria into features, first try to write extraction functions for each criteria)
-	Preparing the data for manual labeling (structuring the collected data, i.e., finding 3 pictures for each asparagus piece, sorting the images in grid, first hands-on approach of a dataset (i.e., sorting on grid), background removal (2/2))

## Manual labeling
-	Creation of manual label app to sort the unlabeled images (generate app & GUI, fusion of single feature extraction algorithms, how well did each feature extraction work?, how much had to be sorted by hand?)
-	Installation of label app (setting up virtual environment (PyQt, sshfs, problems we run into (i.e., not being able to run the app on grid but calling images from external program))
-	User manual for label app (how does it work (i.e., from creating a new file, to uploading images, until saving your output), what does the GUI look like?, what features can you extract?, which features were actually extracted during our hand-sorting process?, what does the output look like?)
-	Sorting criteria for features (explain after which criteria we sorted, what was important to notice?, what was handled less strict?)
-	Sorting outcome (how much did we sort?, how well was it sorted (kappa agreement)?, how well did the sorting go in general (i.e., was it easy to sort?, how long did it take?, what problems were encountered?))

## Neural Networks
-	The dataset (structural information, how many images?, what does it look like?, how to call/work with it?, etc.)
-	Supervised learning (CNN, …)
-	Semi-supervised learning (autoencoder, …)
-	Unsupervised learning (PCA, k-means clustering, …)

## Discussion
-	Outcome of networks (comparison between supervised/semi-supervised/unsupervised approaches)
-	Final result of the project (what went well?, what did not work so good?, what problems did we run into and how did we solve them?, what (technical) restrictions did we encounter?)
-	Further outlook of the project (what can still be done?, how does it contribute to the ANN landscape/open science (e.g., could we publish the dataset)?)
