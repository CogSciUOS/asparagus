# Report chapter 3:&ensp;Preparing the dataset
Here, you can find an overview of how to contribute to chapter 3 of our report, copy & pasted from the report, so you do not need to create a pdf from the .tex files.    
    
> ## 3&ensp;Preparing the dataset 
> &ensp;&ensp;&ensp; Different preprocessing steps of the data and deciding how the data should/will look like.  
> &ensp;&ensp;&ensp; The label app is introduced and the process of manually labelling a part of the data, with a  
> &ensp;&ensp;&ensp; description of the criteria. Finally, the dataset is introduced.  
>  
> &ensp; **Preprocessing steps**    
>     
> &ensp;&ensp;&ensp; First approach to create a dataset (layout) and data augmentation to generate more samples.  
>  
> &ensp;&ensp;&ensp; Automatic feature extraction  `Sophia?`   
> &ensp;&ensp;&ensp; We (tried to) create scripts for: background removal of the collected images, and automatic feature  
> &ensp;&ensp;&ensp; extraction pipeline (including the decision that we first try to sort for features and not labels) for  
> &ensp;&ensp;&ensp; rust, bent, etc. ...  
>  
> &ensp;&ensp;&ensp; Preparation for manual feature extraction  `Maren?`  
> &ensp;&ensp;&ensp; Preparing the images for manual classification to create more labelled data: sorting the pictures in  
> &ensp;&ensp;&ensp; the grid, have 3 pictures per asparagus spear, etc. ...   
>  
> &ensp; **The hand-label app**  `Michael?`  
>  
> &ensp;&ensp;&ensp; Introduction to the script created for manual sorting. Fusion of the feature extraction scripts:   
> &ensp;&ensp;&ensp; What is it? Why did we need it? What was the idea behind it? How does it work? (keep short! it's   
> &ensp;&ensp;&ensp; only the introduction) Do not explain in length here but rather give an idea and refer to README's   
> &ensp;&ensp;&ensp; and to code in GitHub whenever possible.  
>  
> &ensp;&ensp;&ensp; How to install  
> &ensp;&ensp;&ensp; Installation of the app: environment setup, mount points, problems we ran into, etc. ...  
>  
> &ensp;&ensp;&ensp; Operating instructions  
> &ensp;&ensp;&ensp; User manual for the app and introduction to its graphical user interface: What can you find where?  
> &ensp;&ensp;&ensp; (include one example picture of GUI), Step-by-step guideline through loading pictures, creating   
> &ensp;&ensp;&ensp; a .csv file, and how to sort one picture.  
>  
> &ensp;&ensp;&ensp; Performance  
> &ensp;&ensp;&ensp; Results and general performance of the app: How well did the feature extraction work? How much features   
> &ensp;&ensp;&ensp; had to be labelled by hand? What is the output of the app?  
>  
> &ensp; **Manual labeling**
>  
> &ensp;&ensp;&ensp; Sorting criteria  `Josefine`  
> &ensp;&ensp;&ensp; The criteria explained in detail for the hand-labelling of the features with the app (including example   
> &ensp;&ensp;&ensp; pictures). What are expected difficulties we might encounter?  
>  
> &ensp;&ensp;&ensp; Sorting outcome  `Malin?`  
> &ensp;&ensp;&ensp; The process and the results of the sorting: How much did we sort? How well did the sorting work in general  
> &ensp;&ensp;&ensp; (i.e., was it easy to sort? how long did it take? what problems were encountered?)? How accurately did   
> &ensp;&ensp;&ensp; we sort as a group? (i.e., Kappa Agreement)  
>  
> &ensp; **The asparagus dataset**  `Richard?`  
>  
> &ensp;&ensp;&ensp; Different datasets `Sophia?`  
> &ensp;&ensp;&ensp; Structural information on the datasets: What do they look like? How big are they (labelled vs unlabelled   
> &ensp;&ensp;&ensp; samples)? Which were criteria for throwing out data? (maybe have an overview picture with all relevant   
> &ensp;&ensp;&ensp; information on one glance  
>  
> &ensp;&ensp;&ensp; Challenges  `Sophia, Richard`   
> &ensp;&ensp;&ensp; Problems and challenges during the creation of the datasets: What were the challenges in creating a   
> &ensp;&ensp;&ensp; general dataset? What were challenges in general? How well could we work with the datasets? What was   
> &ensp;&ensp;&ensp; used as training data, validation data, and test data?  
