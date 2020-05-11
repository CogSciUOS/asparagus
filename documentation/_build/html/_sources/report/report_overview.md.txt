# Report overview

This is the overview for the report. The actual report is in the main.tex file in the same folder and constructed from the other .tex files in the Chapters folder. To see the most recent changes in the report, please create a pdf file from the main.tex file or have a look at asparagus-report.pdf. 
    
Task distribution is marked `like this`.  

## 1&ensp;Introduction   
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
  
## 2&ensp;Data acquisition and organization 
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
  
## 3&ensp;Preprocessing and data set creation   
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
>    What is it? Why did we need it? What was the idea behind it? How does it work? What is the output of the app?  
  
&ensp; **Manual labeling**  `Josefine`  
  
&ensp;&ensp;&ensp; Sorting criteria  `Josefine`  
  
>    The criteria explained in detail for the hand-labeling of the features with the app (including example  
>    pictures). What are expected difficulties we might encounter?  
  
&ensp;&ensp;&ensp; Sorting outcome  `Josefine`  
  
>    The process and the results of the sorting: How much did we sort? How well did the sorting work in general:  
>    i.e., was it easy to sort? How long did it take? What problems were encountered?  
  
&ensp;&ensp;&ensp; Agreement measures  `Malin`  
  
>    Theoretical background on a measurement that assesses our sorting agreement.
  
&ensp;&ensp;&ensp; Reliability  `Malin`
  
>    Expanding on how accurately we sorted/how valid our sorting was as a group. Introducing the Kappa Agreement. 
  
&ensp; **The asparagus data set**  `Richard, Sophia` 
  
>    Structural information on the datasets: What do they look like? How big are they (labelled vs unlabelled  
>    samples)? Which were criteria for throwing out data?   
>    Problems and challenges during the creation of the datasets: What were the challenges in creating a  
>    general dataset? What were challenges in general? How well could we work with the datasets? What was  
>    used as training data, validation data, and test data?  
  
## 4&ensp;Classification
&ensp; **Supervised learning**  `Josefine`  
  
&ensp;&ensp;&ensp; Feature Engineering  `Michael`   
&ensp;&ensp;&ensp; Single-label classification  `Josefine`    
&ensp;&ensp;&ensp; Multi-label classification  `Sophia`   
&ensp;&ensp;&ensp; Head-related Feature Network  `Michael`   
&ensp;&ensp;&ensp; From Features to Labels  `Katharina`   
  
&ensp; **Unsupervised learning**  `Malin` 
  
&ensp;&ensp;&ensp; Principal component analysis  `Malin, Maren` 
  
&ensp; **Semi-supervised learning**   `Michael`  
  
&ensp;&ensp;&ensp; Autoencoder  `Michael` 
  
## 5&ensp;Summary  `Maren`
  
## 5&ensp;Discussion  `Malin`
&ensp; **Classification results**  `Malin, Michael`
  
&ensp; **Methodology**  `Malin`
  
&ensp; **Organization**  `Josefine, Richard`
   
## 6&ensp;Conclusion  `Richard`
  
>    Outlook of the project. Contribution to scientific landscape? 
