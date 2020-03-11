# How to run the streamlit app and train/add your model

This manual shows you how to run the streamlit app. It also explains how to train your model based on the csv files and the images using the scripts `code/labelCNN/combine_labels.py`, `code/labelCNN/training.py`. For details please look at the source code. Also, have a look at `/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/katha/` to see what a possible set up looks like or how to use the grid to train your model.


## Clone the github repository

`git clone https://github.com/CogSciUOS/asparagus.git`



## Activate the asparagus_env

You can find the `asparagus_env.yml` file to create an environment in the repository root directory. It is most probably not platform independent, so please use a Linux system (e.g. the university computers). Use the command `conda env create -f asparagus_env.yml` to set up the environment. Then activate the environment by using `conda activate asparagus_env`


### Regarding the requirements

You do not need to use the provided environment. Probably you should be fine if you have installed the following Python packages: `tensorflow h5py numpy pandas scikit-image streamlit`


## Preparation

1. The image paths are assumed to have this structure:
   `${IMAGE_DIR}/0/0/0000_a.jpg`, this way we can more or less reliably read
   them from the csv files.


2. Prepare the labels, so that they are in one file:
   **Look at the easy alternative below step 2!**
   `python combine_labels.py {FOLDER_LABELS} labels.csv`
   This merges all csv files inside the directory labels properly into labels.csv
   You can find the script in the folder `code`. Also have a look at the bash script `merge_labels.sh` to see which folder I used and with which command line arguments I called the script. You find it in `/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/katha/`.

   **Easy alternative to point 2.**:  just use the `labels.csv` file I already created stored at `/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/katha/`.

3. To train, run training.py:
   **Skip this part if you want to use an already trained model (it is as easy as just selecting it in the app):**
   `python training.py labels.csv ${SCRATCH_PROJECT}/preprocessed_images/without_background_pngs/ model_name`
   Where images is the `${IMAGE_DIR}` from step 1 and `labels.csv` the file from step 2 and a `name` for the model.
   Make sure you save your trained model to the folder `code/labelCNN/models`. Also have a look at the bash script `train.sge` to see with which command line arguments I called the script. You find it in `/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/katha/`.

2. To inspect the training process, run
   `tensorboard --logdir logs`
   and open a browser at http://localhost:6006


To inspect the results you can use streamlit: `streamlit run app_from_img_to_categories.py` (for details follow the next steps)


## Navigate to the streamlit app

You can find the app that supplies an end to end solution from a csv file containing manually labeled features and image of asparagus pieces to the prediction of the quality class of the asparagus piece in the `code` folder. It is named `app_from_img_to_categories.py`.
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
The next step is to select the path of the image folder e.g. `/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/preprocessed_images/without_background_pngs/`. If the one you want to specify does not appear in the selection bar, please select None and the app allows you to specify the path manually.

## Select csv file

Select a corresponding csv file e.g. `/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/katha/labels.csv` or the one you created in step 2.


## Looking at the app

Go through the app step by step. The app automatically updates after you made a selection.

## If something is not working

Please **let me know** when you encounter problems or have some suggestions on how to improve it. I would also be very happy about **pull requests** that add new features.

Things you can try on your own when running into problems:
1. refresh
2. check all the check boxes from top to bottom
3. check the source code

### Cuda removal

I had to remove `cudnn` and `cudatoolkit` to run everything without a problem. That is why they are not included in the `asparagus_env`.
If that leads to problems, install the environment and remove the two packages selectively without their dependencies.

## Finally

I hope we will add many cool models and explore our data further :)
Have fun playing around with the app! 


### Port forwarding

Port forwarding is still under active investigation. I will keep the manual updated.