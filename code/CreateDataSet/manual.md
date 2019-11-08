# This file should tell you how we created our

## Owerview  

Old vs. new approach:</br>
https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image </br>
However we don't want to use this since with the introduction of tf.data in r1.4, we can create a batch of images without placeholders and without queues.

Another post, coming with a really noice discription: </br>
https://cs230-stanford.github.io/tensorflow-input-data.html#building-an-image-data-pipeline


Often it is helpful is to read the documentation first! </br> https://www.tensorflow.org/datasets/overview </br> That's the corresponding git project </br>
https://github.com/tensorflow/datasets 

However it is sad/more difficult because we (prabably) will have a super huge dataset, so check out: </br>
https://www.tensorflow.org/datasets/beam_datasets </br>
https://beam.apache.org/ 

</br>

## Get hands on

### install conda ... again
This time it has to be installed at shadow or light within your account.
1. log in at gate
2. log in at PC
3. change directory to home/Downloads or wherevery you have rights
3. hit: `$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
4. hit: `$ sh Miniconda2-latest-Linux-x86_64.sh` 
5. hit: `$ conda update conda` 
7. hit: `$ conda activate dataSet`


### Comment:
Conda only keep track of the environments included in the folder envs inside the anaconda folder. 

There are 2 ways to solve it:
1) edit .condarc file in your home directory
    add at envs_dirs:
    - /[PathToEnv]

2) conda activate [absolutpath + EnvName]

### TODO:
- create conda env not in respo but in validUntil...

