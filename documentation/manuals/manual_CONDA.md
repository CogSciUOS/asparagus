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
    or
     `$ conda activate /net/projects/scratch/winter/valid_until_31_July_2020/asparagus/sharedConda/envs/dataSet/`
 6. Test a tensorflow file on the GPU</br>
    `$ nvidia-smi -l`


Check this link out, if you want to know more about shared installations:
https://docs.anaconda.com/anaconda/install/multi-user/ 