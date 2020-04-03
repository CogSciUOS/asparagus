# Create a customised dataset
This file tells you how to do it!

## Overview  
Old vs. new approach:</br>
https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image </br>
Since with the introduction of tf.data in release r1.4, we can create a batch of images without placeholders and without queues.


Often it is helpful to read the documentation first! </br>https://www.tensorflow.org/datasets/overview </br>That's the corresponding git project </br>https://github.com/tensorflow/datasets 


However it is sad/more difficult because we (probably) will have a super huge dataset, so check out: </br>
https://www.tensorflow.org/datasets/beam_datasets </br>
https://beam.apache.org/ 


Another post, coming with a really nice discription: </br>https://cs230-stanford.github.io/tensorflow-input-data.html#building-an-image-data-pipeline</br>

</br>


# Get hands on - use shared conda installation
To use the shared installation you have to tell linux where it is.
1. Log in
    `ssh ...@gate.ikw.uos.de`
2. Add some lines at the end of your bashrc file: </br>
    `$ nano -Bu ~/.bashrc` </br> 
    Or you could use ranger or vim instead...</br>
    ```
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin/conda' 'shell.bash' '$if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/etc/profile.d/conda.sh" ]; then        . "/net/projects/scratch/winter/        valid_until_31_July_2020/asparagus/sharedConda/etc/profile.d/conda.sh"
        else
        export PATH="/net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    ```


3. For changes to take effect, close and re-open your current shell.</br>
    `$ exit` 
4. Log in again. If you want to use a single GPU log in to shadow or light.
    You can test it with:
    `$ nvidia-smi`
4. If you prefer that condas base environment is not activated on startup,</br>
    set the auto_activate_base parameter to false:</br>
    `$ conda config --set auto_activate_base false`
5. After Log in hit: </br>
    `$ conda activate dataSet`
6. Test a tensorflow file on the GPU</br>
    `$ nvidia-smi -l`


Check this link out, if you want to know more about shared installations:
https://docs.anaconda.com/anaconda/install/multi-user/ 


# TODO:
- wirte code to create dataset! 
    - try with MNIST, add / change labels afterwards
    - **make OUR dataset**
