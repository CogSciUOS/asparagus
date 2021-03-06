# Friday, May 03, 2019 (put topic here)

Attendants: Ulf, Richard, Malin, Luana, Maren, Josefine, Katharina, Sophia, Thomas, Michael 

## Summary of last week's work

## Classification decision tree: 
* Firs decision tree is created:
*   "Hollow" is missing.

## Research and Library
* New papers have been added to the repository. 

## Preprocessing and naming conventions
* Cut images, give each one a label 
* Naming convention (name, ID, date, time) 
* Put each peace in a new folder with: id_number_specific number
* Input of image preprocessing: e.g. G01-190411-092055-873_F00.bmp (contains date, batch number and position on band)
* Output of image preprocessing: e.g. 0_0.jpg = id_number, where 'id' is the newly assigned ID and 'number' takes on values from 0 to 2, indicating the position on the band. The mapping of ID to original name is maintained, because the preprocessor creates a csv file that stores for each id what the original name was.
* Output of image augmentation: e.g. rot2_0_0.jpg = pre#_id_number, where 'pre' is a 3-letter code to identify the transformation that was applied and # is the index (e.g. rot=rotation creates 3 images, so rot2 would be the third one of these, where the image was rotated to the right.)
* Transformation codes:
    * rot = rotation 
    * mir = mirroring (along y axis)
    * tra = translation (by some random number of pixels)
* This process is recursive. Transformations are applied in layers, so the output of one augmentation is the input for the next augmentation. e.g. tra1_mir0_rot0_3_1.jpg is id=3, second image, first rotation, first mirroring, second translation. Rotation was applied first, translation last. 
* We can create a lot of labelled artificial data from a small source.

* Integrate preprocessing into the pipeline (Ulf)
* Think of a database/table to store the characteristics of a Pic-ID.

## Label software
* Give the created images a label by adding a prefix.
* We should continue to build an interface that makes it easy for us to label data ourselves.

## Service
* The service was tested on the machine. However, the machine will output an error because it is trying to access the directly moving images.
* It is possible to start and stop the service by pressing the Windows key + enter "Service" and open the window. Search for Directory Check and use the GUI.


## Team Viewer
* plan to manually collect and move images using the Google Calendar.

## During the meeting
* We had a discussion about how to use git. It will be finished.


