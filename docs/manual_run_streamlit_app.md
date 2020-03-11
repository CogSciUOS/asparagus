# How to run the streamlit app and train/add your model

This manual shows you how to run the streamlit app. It also explains how to train your model based on the csv files and the images using the scripts `code/labelCNN/combine_labels.py`, `code/labelCNN/training.py`. For details please look at the source code. Also, have a look at `/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/katha/` to see what a possible set up looks like or how to use the grid to train your model.


## Clone the github repository

`git clone https://github.com/CogSciUOS/asparagus.git`



## Activate the asparagus_env

You can find the `asparagus_env.yml` file to create an environment in the root directory. It is most probably not platform independent, so please use a Linux system (e.g. the university computers). Use the command `conda env create -f environment.yml` to set up the environment. Then activate the environment by using `conda activate asparagus_env`


### Regarding the requirements

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



To inspect the results you can use streamlit: `streamlit run app_from_img_to_categories.py` (for details follow the next steps)


## Navigate to the streamlit app

You can find the app that supplies an end to end solution in the `code` folder. It is named `app_from_img_to_categories.py`.
In order to run the app type `streamlit run app_from_img_to_categories.py` in the terminal.

## Open browser at the provided URL

In the terminal you will find output similar to this:
```
  You can now view your Streamlit app in your browser.

  Network URL: http://192.168.0.8:8501
  External URL: http://31.17.252.51:8501
```
Copy the Network URL into the address field of the browser of your choice.


## Select the path to image folder

Now you should see that the app is running and greets you with "Asparagus label prediction".


The next step is to select the path of the image folder. As we labeled on the 
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