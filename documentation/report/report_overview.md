# Report overview

This is the overview for the Report. The actual report is in the main.tex file in the same folder and constructed from the other .tex files in the Chapters folder. To see the most recent changes in the report, please pull in the origin/Report branch and create a pdf file from the main.tex file. 
    
Task distribution is marked `like this`.  

## 1&ensp;Introduction `Josefine`  
&ensp; **The project**  `Josefine`
  
>    Rough summary of the idea of the project.     
>    How did the idea come up?   
>    What is the project about?   
  
&ensp; **Background on computer vision based classification tasks**  `Sophia`
  
>    Short introduction into the field of computer vision and short introduction to artificial neural networks.  
>    Prospects and challenges of both on a general basis.  
>    This part is kept very brief.  
  
&ensp; **Background on sorting asparagus**  `Josefine`
  
>    What is the main focus during the sorting process, i.e. why do you need to sort asparagus and in which  
>    classes do you sort it?  
>    What problems and challenges will be met - including the difference of challenge for humans vs. machines.  
>    Including the decision tree for the labels as sorted with Silvan’s machine.  
  
&ensp; **Expected outcome vs. actual outcome of the project**  `Malin`
  
>    What did we hope to achieve with the project and, in contrast to that, what was the actual outcome?  
>    What challenges did we expect and what results did we aim for?  
>    Which challenges did we not expect and what did we do to overcome them?  
>    Did we actually solve them?  
  
## 2&ensp;Data acquisition and organization  `Josefine`
&ensp; **Timetable (roadmap) of the project**  `Josefine`  
  
>    A (visual) overview of the project's course during the year.  
>    How we planned the project vs. how the actual timetable of the project looked like.  
  
&ensp; **Organisation of the study group**  `Richard`
  
&ensp;&ensp;&ensp; Communication  `Richard`  
  
>    Deciding on a communication platform and handling it: organization in general, team meetings,  
>    Asana, Telegram, and any other virtual form of communication  
  
&ensp;&ensp;&ensp; Teamwork  `Richard`  
  
>    The experience of working in a team and organising/distributing tasks in a group:  
>    task distribution, cooperation on tasks, GitHub: organising our project with Git,  
>    Grid: working together with the IKW data store, other teamwork related experiences  
>    (could we integrate the strengths and weaknesses of the single team members) ...  
  
&ensp; **Data collection**  `Josefine, Richard`
  
>    How we collected the data and what the data looked like.  
>    Here, the process of driving to Rheine and collecting the data with an external hard-drive is described.  
>    Understanding the sorting machine and its output.  
>    First problems that had to be resolved: labelled vs. unlabelled data ( -> running pre-sorted pieces  
>    through the sorting machine, did not (completely) resolve our problem), saving the data manually  
>    on an external harddrive ( -> solved by building a script for data transfer, and Teamviewer sessions)  
  
&ensp; **Literature research**  `Josefine`  
  
>    Previous literature research concerning food classification and handling unlabelled data.  
>    Searching for background literature close to our project, e.g. automatic CV-based sorting of other  
>    food products.  
>    Re-reading on potential ANN structures that we could use for sorting.  
>    Could we rely on a certain paper/process? Did it work?  
  
## 3&ensp;The dataset `Josefine`  
&ensp; **Preprocessing steps**  `Sophia` 
  
>    First approach to create a dataset (layout) and data augmentation to generate more samples.  
>    Preparation for manual feature extraction, background removal, etc. ...    
>    Preparing the images for manual classification to create more labelled data: sorting the pictures in  
>    the grid, have 3 pictures per asparagus spear, etc. ...  
  
&ensp; **Automatic feature extraction**  `Sophia, Michael`  
  
>    We (tried to) create scripts for an automatic feature extraction pipeline  
>    (including the decision to sort for features and not labels) for   
>    rust, bent, etc. ...  (all features)  
  
&ensp; **The hand-label app**  `Michael`
  
>    Introduction to the script created for manual sorting. Fusion of the feature extraction scripts:  
>    What is it? Why did we need it? What was the idea behind it? How does it work? (keep short! it's  
>    only the introduction) Do not explain in length here but rather give an idea and refer to README's  
>    and to code in GitHub whenever possible.  
  
&ensp;&ensp;&ensp; How to install `Michael`  
  
>    Installation of the app: environment setup, mount points, problems we ran into, etc. ...  
  
&ensp;&ensp;&ensp; Operation instructions `Michael`  
  
>    User manual for the app and introduction to its graphical user interface: What can you find where?  
>    (include one example picture of GUI), Step-by-step guideline through loading pictures, creating  
>    a .csv file, and how to sort one picture.  
  
&ensp;&ensp;&ensp; Performance  `Michael`  
  
>    Results and general performance of the app: How well did the feature extraction work? How much features  
>    had to be labelled by hand? What is the output of the app?  
  
&ensp; **Manual labelling**  `Josefine`  
  
&ensp;&ensp;&ensp; Sorting criteria  `Josefine`  
  
>    The criteria explained in detail for the hand-labelling of the features with the app (including example  
>    pictures). What are expected difficulties we might encounter?  
  
&ensp;&ensp;&ensp; Sorting outcome  `Josefine`  
  
>    The process and the results of the sorting: How much did we sort? How well did the sorting work in general  
>    (i.e., was it easy to sort? how long did it take? what problems were encountered?)? How accurately did  
>    we sort as a group? (i.e., Kappa Agreement)  
  
&ensp;&ensp;&ensp; Agreement measures  `Malin`  
  
>    Theoretical background on a measurement that assesses our sorting agreement.
  
&ensp;&ensp;&ensp; Validity  `Malin`
  
>    Expanding on how accurately we sorted/how valid our sorting was as a group. Introducing the Kappa Agreement. 
  
&ensp; **The asparagus dataset**  `Richard?` 
  
&ensp;&ensp;&ensp; Different datasets  `Sophia, Richard`  
  
>    Structural information on the datasets: What do they look like? How big are they (labelled vs unlabelled  
>    samples)? Which were criteria for throwing out data? (maybe have an overview picture with all relevant  
>    information on one glance  
  
&ensp;&ensp;&ensp; Challenges   `Richard, Sophia`
  
>    Problems and challenges during the creation of the datasets: What were the challenges in creating a  
>    general dataset? What were challenges in general? How well could we work with the datasets? What was  
>    used as training data, validation data, and test data?  
  
## 4&ensp;Classification `Malin`
&ensp; **Supervised learning**  `Josefine`  
  
&ensp;&ensp;&ensp; Single-label classification  `Josefine`  
&ensp;&ensp;&ensp; Multi-label classification  `Sophia` 
  
&ensp; **Semi-supervised learning**   `Michael, Richard`
  
&ensp;&ensp;&ensp; Autoencoder  `Michael` 
  
&ensp; **Unsupervised learning**  `Malin` 
  
&ensp;&ensp;&ensp; Principal component analysis  `Malin, Maren` 
  
&ensp; **From feature to label**  `Josefine` 

## 5&ensp;Discussion
&ensp; **Comparison of classification approaches**  
  
&ensp;&ensp;&ensp; Comparing architectures  
&ensp;&ensp;&ensp; Comparing results

  
&ensp; **Final result of the project**  
  
&ensp;&ensp;&ensp; Scientific results  
&ensp;&ensp;&ensp; Organization   

## 6&ensp;Conclusion  
  
&ensp; **Summary**  
    
&ensp; **Next steps**   `Richard`
  
&ensp;&ensp;&ensp; Outlook of the project  
&ensp;&ensp;&ensp; Contribution to scientific landscape  
