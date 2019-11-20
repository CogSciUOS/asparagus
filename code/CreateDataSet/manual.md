# This file tells you how to create your dataset

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
4. hit: `$ sh Miniconda3-latest-Linux-x86_64.sh` 
4. if you get asked if conda shoud be init hit `yes`
4. hit exit as often as you can and restart the seassion
5. hit: `$ conda update conda` 
5. hit: `conda config --set auto_activate_base false`
7. hit: `$ conda activate dataSet`

*TODO:* Create Bash-Script: https://www.taniarascia.com/how-to-create-and-use-bash-scripts/ 




2.Solution:
current path to the enviroment is:
`/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/DataEnv/`

Add an alias to activate your env: </br>
sudo nano -Bu ~/.bashrc </br>
Or you could use ranger or vim instead... </br>
add this line at the end: 
`'alias dataEnv='conda activate /net/projects/scratch/summer/valid_until_31_January_2020/asparagus/DataEnv'`



### Comment:
Conda only keep track of the environments included in the folder envs inside the anaconda folder. 

There are 2 ways to solve it:
1) edit .condarc file in your home directory
    add at envs_dirs:
    - /[PathToEnv]

2) conda activate [absolutpath + EnvName]


### TODO:
- update conda env
- write bash script?!
- wirte code to create dataset! 
    - try with MNIST, add / change labels afterwords
- make dataset available -> new manual




CurrentProblem: </br>
Datasorrage exeeded - packages are insalled in home/r/rruppel/miniconda3 ... not at the env position
Solution: https://docs.anaconda.com/anaconda/install/multi-user/ install conda for everyone? ... however i am not a admin, cant create groups and so on... 

erklärungsansatz: es gibt den path PATH=$PATH:$HOME/.local/bin:$HOME/bin in dem file ~/.bash_profile, wo der path angegeben ist.

https://stackoverflow.com/questions/54429210/how-do-i-prevent-conda-from-activating-the-base-environment-by-default 



rruppel@light:~/Downloads$ sh Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 4.7.12

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>
===================================
Miniconda End User License Agreement
===================================

[...]

Do you accept the license terms? [yes|no]
[no] >>>
Please answer 'yes' or 'no':'
>>> yes

Miniconda3 will now be installed into this location:
/home/student/r/rruppel/miniconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home/student/r/rruppel/miniconda3] >>> //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/
PREFIX=//net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation

  added / updated specs:
    - _libgcc_mutex==0.1=main
    - asn1crypto==1.2.0=py37_0
    - ca-certificates==2019.10.16=0
    - certifi==2019.9.11=py37_0
    - cffi==1.13.0=py37h2e261b9_0
    - chardet==3.0.4=py37_1003
    - conda-package-handling==1.6.0=py37h7b6447c_0
    - conda==4.7.12=py37_0
    - cryptography==2.8=py37h1ba5d50_0
    - idna==2.8=py37_0
    - libedit==3.1.20181209=hc058e9b_0
    - libffi==3.2.1=hd88cf55_4
    - libgcc-ng==9.1.0=hdf63c60_0
    - libstdcxx-ng==9.1.0=hdf63c60_0
    - ncurses==6.1=he6710b0_1
    - openssl==1.1.1d=h7b6447c_3
    - pip==19.3.1=py37_0
    - pycosat==0.6.3=py37h14c3975_0
    - pycparser==2.19=py37_0
    - pyopenssl==19.0.0=py37_0
    - pysocks==1.7.1=py37_0
    - python==3.7.4=h265db76_1
    - readline==7.0=h7b6447c_5
    - requests==2.22.0=py37_0
    - ruamel_yaml==0.15.46=py37h14c3975_0
    - setuptools==41.4.0=py37_0
    - six==1.12.0=py37_0
    - sqlite==3.30.0=h7b6447c_0
    - tk==8.6.8=hbc83047_0
    - tqdm==4.36.1=py_0
    - urllib3==1.24.2=py37_0
    - wheel==0.33.6=py37_0
    - xz==5.2.4=h14c3975_4
    - yaml==0.1.7=had09818_2
    - zlib==1.2.11=h7b6447c_3


The following NEW packages will be INSTALLED:

  _libgcc_mutex      pkgs/main/linux-64::_libgcc_mutex-0.1-main
  asn1crypto         pkgs/main/linux-64::asn1crypto-1.2.0-py37_0
  ca-certificates    pkgs/main/linux-64::ca-certificates-2019.10.16-0
  certifi            pkgs/main/linux-64::certifi-2019.9.11-py37_0
  cffi               pkgs/main/linux-64::cffi-1.13.0-py37h2e261b9_0
  chardet            pkgs/main/linux-64::chardet-3.0.4-py37_1003
  conda              pkgs/main/linux-64::conda-4.7.12-py37_0
  conda-package-han~ pkgs/main/linux-64::conda-package-handling-1.6.0-py37h7b6447c_0
  cryptography       pkgs/main/linux-64::cryptography-2.8-py37h1ba5d50_0
  idna               pkgs/main/linux-64::idna-2.8-py37_0
  libedit            pkgs/main/linux-64::libedit-3.1.20181209-hc058e9b_0
  libffi             pkgs/main/linux-64::libffi-3.2.1-hd88cf55_4
  libgcc-ng          pkgs/main/linux-64::libgcc-ng-9.1.0-hdf63c60_0
  libstdcxx-ng       pkgs/main/linux-64::libstdcxx-ng-9.1.0-hdf63c60_0
  ncurses            pkgs/main/linux-64::ncurses-6.1-he6710b0_1
  openssl            pkgs/main/linux-64::openssl-1.1.1d-h7b6447c_3
  pip                pkgs/main/linux-64::pip-19.3.1-py37_0
  pycosat            pkgs/main/linux-64::pycosat-0.6.3-py37h14c3975_0
  pycparser          pkgs/main/linux-64::pycparser-2.19-py37_0
  pyopenssl          pkgs/main/linux-64::pyopenssl-19.0.0-py37_0
  pysocks            pkgs/main/linux-64::pysocks-1.7.1-py37_0
  python             pkgs/main/linux-64::python-3.7.4-h265db76_1
  readline           pkgs/main/linux-64::readline-7.0-h7b6447c_5
  requests           pkgs/main/linux-64::requests-2.22.0-py37_0
  ruamel_yaml        pkgs/main/linux-64::ruamel_yaml-0.15.46-py37h14c3975_0
  setuptools         pkgs/main/linux-64::setuptools-41.4.0-py37_0
  six                pkgs/main/linux-64::six-1.12.0-py37_0
  sqlite             pkgs/main/linux-64::sqlite-3.30.0-h7b6447c_0
  tk                 pkgs/main/linux-64::tk-8.6.8-hbc83047_0
  tqdm               pkgs/main/noarch::tqdm-4.36.1-py_0
  urllib3            pkgs/main/linux-64::urllib3-1.24.2-py37_0
  wheel              pkgs/main/linux-64::wheel-0.33.6-py37_0
  xz                 pkgs/main/linux-64::xz-5.2.4-h14c3975_4
  yaml               pkgs/main/linux-64::yaml-0.1.7-had09818_2
  zlib               pkgs/main/linux-64::zlib-1.2.11-h7b6447c_3


Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/condabin/conda
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/bin/conda
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/bin/conda-env
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/bin/activate
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/bin/deactivate
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/etc/profile.d/conda.sh
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/etc/fish/conf.d/conda.fish
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/shell/condabin/Conda.psm1
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/shell/condabin/conda-hook.ps1
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/lib/python3.7/site-packages/xontrib/conda.xsh
no change     //net/projects/scratch/summer/valid_until_31_January_2020/asparagus/condaInstallation/etc/profile.d/conda.csh
**modified      /home/student/r/rruppel/.bashrc**

==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup,
   set the auto_activate_base parameter to false:

conda config --set auto_activate_base false

Thank you for installing Miniconda3!




~/.bashrc  got add:


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

