# How to run the end to end streamlit app and add your model

This manual shows you how to run the streamlit app. It also explains how to train your model based on the csv files and the images using the script `code/labelCNN/training.py`. For details please look at the source code.


## Clone the github repository

`git clone https://github.com/CogSciUOS/asparagus.git`



## Activate the asparagus_env



### Requirements

You do not need to use the provided environment. Probably you should be fine if you have installed the following Python packages: `tensorflow h5py numpy pandas scikit-image streamlit`


## Preparation

1. The image paths are assumed to have this structure:
   `${IMAGE_DIR}/0/0/0000_a.jpg`, this way we can more or less reliably read
   them from the csv files.

2. Prepare the labels, so that they are in one file:
   `python combine_labels.py labels labels.csv`
   This merges all csv files inside the directory labels properly into labels.csv


**Skip this part if you want to use an already trained model (it is as easy as just selecting it in the app):**

3. To train, run training.py:
   `python `
   Where images is the `${IMAGE_DIR}` from step 2 and labels.csv the file from step 3.
   Make sure you save your trained model to the folder `here`.

4. To inspect the training process, run
   `tensorboard --logdir logs`
   and open a browser at http://localhost:6006



To inspect the results you can use streamlit: `streamlit run app.py` (for details follow the next steps)


## Navigate to the streamlit app


## Open browser at 


## Select the path to image folder

if None  you can enter path directly in the app

## Select csv file



## Looking at the app


## When something is not working

This is my first streamlit app. So although I tried my best to structure the app clearly and make it faster by caching some functions, there is room for improvement. Please **let me know** when you encounter problems or have some suggestions on how to improve it. I would also be very happy about **pull requests** that add new features.

Things you can try on your own when running into problems:
1. refresh
2. check all the checkboxes from top to bottom
3. check the source code

## Finally

I hope we will add many cool models and explore our data further :)
Have fun playing around with the app! 


### Port forwarding

I really do not know why the port forwarding does not work reliably. If I can figure it out, I will update this manual.