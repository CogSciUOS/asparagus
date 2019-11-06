# This file should tell you how we created our

## Owerview  

Old vs. new approach:
https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image 
However we don't want to use this since with the introduction of tf.data in r1.4, we can create a batch of images without placeholders and without queues.

another post, coming with a really noice discription
https://cs230-stanford.github.io/tensorflow-input-data.html#building-an-image-data-pipeline


Often it is helpful is to read the documentation first!
https://www.tensorflow.org/datasets/overview 
That's the corresponding git project
https://github.com/tensorflow/datasets 

However it is sad/more difficult because we (prabably) will have a super huge dataset
So check that: 
https://www.tensorflow.org/datasets/beam_datasets 
https://beam.apache.org/ 


## Get hands on

**TODO:**
- create conda env not in respo but in validUntil...



comment:
Conda only keep track of the environments included in the folder envs inside the anaconda folder. 

There are 2 ways to solve it:
1) edit .condarc file in your home directory
    add at envs_dirs:
    - /[PathToEnv]

2) conda activate [absolutpath + EnvName]

Working with the env:
**Please help to keep this requirements file up to date!**
The file should be stored in this (Create_TF_Dataset) folder in the repos

