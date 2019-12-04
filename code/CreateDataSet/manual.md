# Create a customised dataset
This file tells you how to do

# Owerview  
Old vs. new approach:</br>
https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image </br>
Since with the introduction of tf.data in release r1.4, we can create a batch of images without placeholders and without queues.


Often it is helpful is to read the documentation first! </br>https://www.tensorflow.org/datasets/overview </br>That's the corresponding git project </br>https://github.com/tensorflow/datasets 


However it is sad/more difficult because we (prabably) will have a super huge dataset, so check out: </br>
https://www.tensorflow.org/datasets/beam_datasets </br>
https://beam.apache.org/ 


Another post, coming with a really noice discription: </br>https://cs230-stanford.github.io/tensorflow-input-data.html#building-an-image-data-pipeline</br>

</br>


# Get hands on - use shared conda installation
To use the shared the installation you have to tell linux where it is.
1. Log in
    `ssh ...@gate.ikw.uos.de`
2. Add some lines at the end of your bashrc file: </br>
    `$ nano -Bu ~/.bashrc` </br> Or you could use ranger or vim instead...</br>
    ```
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('//net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "//net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/etc/profile.d/conda.sh" ]; then
            . "//net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/etc/profile.d/conda.sh"
        else
            export PATH="//net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    ```
3. For changes to take effect, close and re-open your current shell.</br>
    `$ exit` 
4. If you'd prefer that conda's base environment not be activated on startup,</br>
    set the auto_activate_base parameter to false:</br>
    `conda config --set auto_activate_base false`
5. After Log in hit: </br>
    `$ conda activate dataSet`

If you want to know more about shared installations:
https://docs.anaconda.com/anaconda/install/multi-user/ 

### TODO:
- wirte code to create dataset! 
    - try with MNIST, add / change labels afterwords
    - **make OUR dataset**