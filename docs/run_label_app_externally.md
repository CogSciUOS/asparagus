* a copy - paste of the process on my laptop

Last login: Mon Sep  9 10:43:46 on ttys001
**(base) 83adbada:~ Malin$ ssh mspaniol@gate.ikw.uos.de **
mspaniol@gate.ikw.uos.de's password: 
Welcome to Ubuntu 12.04.5 LTS (GNU/Linux 3.13.0-117-generic x86_64)

 * Documentation:  https://help.ubuntu.com/
This Ubuntu 12.04 LTS system is past its End of Life, and is no longer
receiving security updates.  To protect the integrity of this system, it’s
critical that you enable Extended Security Maintenance updates:
 * https://www.ubuntu.com/esm

This computer belongs to the Institute of Cognitive Science 
at the University of Osnabrueck.
                
The usage of this computer as well as all its services
is subject to the terms of use provided at the following 
URL: https://doc.ikw.uni-osnabrueck.de/policy/user
  
If you do not agree with these terms please do log out!

 Yours sincerely,
   the IKW-Admins (3362, 50/E14)

Last login: Mon Sep  9 12:20:51 2019 from 83adbada.funky.uni-osnabrueck.de
**(base) mspaniol@gate:~$ ssh -X light ** - light and shadow are the computers in our room
mspaniol@light's password: 
Welcome to Ubuntu 16.04.6 LTS (GNU/Linux 4.4.0-159-generic x86_64)

95 packages can be updated.
1 update is a security update.

*** System restart required ***
This computer belongs to the Institute of Cognitive Science 
at the University of Osnabrueck.
                
The usage of this computer as well as all its services
is subject to the terms of use provided at the following 
URL: https://doc.ikw.uni-osnabrueck.de/policy/user
  
If you do not agree with these terms please do log out!

 Yours sincerely,
   the IKW-Admins (3362, 50/E14)

Last login: Mon Sep  9 15:13:38 2019 from shadow.cv.uni-osnabrueck.de
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus$ lsus
GitProject  Images  backup_1
** (base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus$ ls **
GitProject  Images  backup_1
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus$ cd GitProject
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/GitProject$ cd asparagus
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/GitProject/asparagus$ cd code
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/GitProject/asparagus/code$ cd hand_label_assistant
**(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/GitProject/asparagus/code/hand_label_assistant$ python main.py **
qt.qpa.xcb: could not connect to display 
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.

Aborted


Normally, this should be enough to run the app, maybe you still need to install some packages 

(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/GitProject/asparagus/code/hand_label_assistant$ pip install pyqt5
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/GitProject/asparagus/code/hand_label_assistant$ pip install numpy
(base) mspaniol@light:/net/projects/scratch/summer/valid_until_31_January_2020/asparagus/GitProject/asparagus/code/hand_label_assistant$ pip install imageio
