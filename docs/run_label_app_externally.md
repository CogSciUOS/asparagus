# Manual: running label-app on uni network


A copy - paste of the process on my laptop

log into uni server network: 
```
(base) 83adbaa6:~ Malin$ ssh -Y mspaniol@gate.ikw.uos.de
```

you want to log onto a specific computer in the network, such as light (check if you have the permission, else, talk to Ulf):
```
(base) mspaniol@gate:~$ ssh -Y light
```

now get to our github directory, if you already made yourself an alias to get there quickly, do that!
```
(base) mspaniol@light:cd /net/projects/scratch/summer/valid_until_31_January_2020/asparagus
```

create a personal folder with your name in the asparagus folder (you can already see katha, josefine and malin there)
```
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus$ mkdir [YOUR_NAME]
```

go into your folder and clone the github repository there:
```
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/[YOUR_NAME]$ git clone https://github.com/CogSciUOS/asparagus.git
```
in your cloned repository, you should go into asparagus and switch to the branch hand_label_assistant
```
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/malin/asparagus$ git checkout hand_label_app
```

start the virtual environment (If you end up with problems here, look at the pipenv documentation and make sure pipenv is installed via pip)
```
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/malin/asparagus$ pipenv install
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/malin/asparagus$ pipenv shell
```

 then you want to go into hand_label_assistant:
```
(asparagus) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/malin/asparagus$ cd code/hand_label_assistant
```
```
(asparagus) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/malin/asparagus/code/hand_label_assistant $ python main.py
```
