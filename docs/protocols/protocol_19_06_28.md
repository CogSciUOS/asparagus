# Friday, June 28th, 2019 (new repository, data storage, preprocessing plan, holidays)

Attendants: Michael, Katha, Luana, Ulf, Axel, Malin 

## New repository 
- there is a new repository (forked from the old one) with a new folder structure, better overview (Katha presented it) https://github.com/CogSciUOS/asparagus 
  - from now on: edit and work on this repository and make sure to use the correct documentation
- our travis ci link: _https://travis-ci.org/CogSciUOS/asparagus 

## Organisational issues proposed by Ulf: 
- lehrevaluation should be done until next wednesday! axel sent everyone an email
- do we know how to access all our data? --> we should organise it (now in scratch folder, unorganised) 
  - our current folder is valid until January 2020
  - make sure if you create file there: that it is writable by everyone of group 
    - some command help: chmod -R g+w file /directory
          - -R given recursive naming for all subdirectories g+w file 
          - man (for manual) chmod 
          - [chmod g+s ]
          - df -h 
          -  or search online e.g. “chmod - I want to change name of group …. “
 
- make schedule for the holidays, everyone should enter when they are in Osnabrück, so that we know when it makes sense to meet (suggestion: only a few big meetings, and then people meet in smaller groups) 
- the ones receiving responders - pick them up

- make time plan, deadlines, see if the resources are there to try superresolution approach (for head images of Rheine machine, by having good head pictures from Querdel)

## Work to do
- organise pictures on storage server - Maren (until next week)
  (make a copy, then start to organise everything, renaming …. maybe split into subfolders 
  - suggestion: make a deep copy (read only! as backup)
  - then: one folder: labeled images, another folders with subfolders for unlabelled images (10.000 images per folder)
  - one preprocessing folder with subfolders (such as unlabeld folders) (we decided: if the middle asparagus does not lie in the cut window, we throw the image away)
  
- Michael and Thomas have to talk how the preporcessing is done ( with their code which already exists? )
  - we want to give the labels based on the asparagus pieces, not based on image —> so we have to get all three images belonging to one piece together  (start the foldering order by beginning of july)
 
- Michael: labelling app should be ready by Mid July 
-  Use label app / make hand labelling: August 
       
- still open: put a few images into program and see if we can extract features! - sophia wanted to do it 
