Friday 26.04.19
50/E04
8.30h-10h

Sophia, Maren, Luana, Malin, Thomas, Katharina, Josefine, Richard, Michael, Ulf

# Excursion Today

Organisatorial Stuff
-	meeting at Rheine at around ~15h to visit the asparagus sorting machine
-	we can go there by car (3) or train

What do we want to do?
-	collect more data (take hard drive with us)
-	trying to install Richard's tool
-	talk to Sophia’s brother again to explain
-	maybe try to get some labelled data
-	we still need the list of categories (which could be acquired once we’re there)

# Project Management Tools

Presentation by Luana:
-	Trello	
-	ClickUp
-	Asana
-	Google Drive		

__Vote for tool: Asana__ (no strong opinion)

Github
-	also has a project management system
-	we would not need another user account
-	university server can be used for file/image storage
  -	folder name “Asparagus”
  -	pathway: net/projects/scratch/summer/valid_until_31_January_2020/asparagus
  -	when creating a folder, give access to other group members

__New vote: Github__
-	if it is not to our liking after two weeks we will use Asana

Github Project “Asparagus
-	we will use other projects for (smaller) tasks

# Paper Research and Library
 
Presentation by Katharina and Josefine
- introduction of a library system for paper research
  - add short info (title, author, year, helpfulness etc) about any paper you read into it
  - the info categories are still up for debate
- short summary of 5 papers
  - Donis-González and Guyer (2016) (asparagus sections)
  - Diaz et al. (2004) (table olives)
  - Pedreschi, Mery, and Marique (2016) (potatoes)
  - Kılıç et al. (2007) (beans)
  - Mery, Pedreschi, and Soto (2013) (general framework)
  
What papers might be interesting to look for?
-	image acquisition
  - illumination?
-	small subset of labelled data
  - key word: semisupervised learning
  - autoencoders

Other papers found (which could be added to the library?)
- flower classification (Luana)
- non-labelled images (Maren)
- feature extraction (Malin)

Discussion: Usefulness of having a collection of papers
- keeps it organized to avoid double reading
- able to look up key words/papers for later problems
- have a guideline
- should not loose ourselves in too many details here

# Open discussion

Image acquisition
-	little less than two months left for image acquisition
-	might be most important part right now
  – illumination should be fine (no further work on that needed)

3 remarks by Ulf
-	looking at literature is always good
-	give pros and cons for your rating of a paper
-	now we should focus on data acquisition (not going too deep into methods right now)
-	for future: should definitely have look at working with *unbalanced datasets*, because most other papers work with balanced data sets

Vote for library system of papers
-	keep it (8)
-	not useful enough (1)

Infos about program on classification machine
- can store features of images, date etc.
- on screen it seems it did well with borders etc.
- the question is: why is the machine so bad, if the features look so well
  - if the features are bad we need new ones
  - but if the features are good it woud make the task easier because we could use the machine's features
  - most difficult feature seemed the colouring
   
-	for future: if we have labelled data, it would be good to have a folder for each category
  - naming conventions (to avoid double naming/overcomplicated names)

# Preparations & next session

To Do:
-	update about "playing" with pictures (*Thomas*)
  - cut image into 3 parts to have one piece of asparagus per picture
  - filter out background (find a bounding box around asparagus), i.e. have distinct bright background (to have no confusion with dirt/purple asparagus/dark, rusty stains)
  - work on preprocessing steps
  - reducing file size (right now it is ~5 GB)
  - goal: have images that all look the same (right now there are 1-3 asparagus per picture) and are small enough
   
-	have a look at Grid again and maybe also have a small Github "tutorial" meeting (*Katharina*)
  -	you can have a look at the code of the service on github
  -	meeting after regular Friday morning session (03.05.19, 10h)

-	start implementation (*Malin, Richard, Maren*)
  - reading up on implementing pictures/preprocessing papers
  -	use algorithms learned in lectures

-	try to figure out how the machine program works to understand its feature selection (*Sophia*)
  - which features are extracted?
  
-	think about a guideline for the (manual) picture classification
  - create a cheat sheet with sorting instructions (step-by-step) (*Josefine*)
   - best would be same categories as the machine uses
  - look into having code for sorting pictures in different categories (to make work easy) (*Michael*)
  
-	look at unbalanced data sets (data set augmentation) and maybe semisupervised learning (*Luana*)

Topics for next week:
- github: branching, naming conventions
