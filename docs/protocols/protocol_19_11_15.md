Protocol from 15.11.2019  
# Kappa Agreement, Image labeling, First WS Meeting   
Attendants: Ulf, Axel, Sophia, Richard, Michael, Malin, Josefine

## Intro

What we’ve done since the semester started
-	We are working on a kappa agreement
-	Everyone installed the feature extraction app and labeled min. 200 images
-	Sophia: Can we go from features we sort by hand with decision rules back to labels/classes Silvan gave the asparagus?
-	What’s still not always working are thresholds for thickness and width (pixel to cm), we need better ‘borders’
-	Our sorting does not necessarily produce ideal prototypes; however, outliers (=one or two wrongly classified spears) aren’t grave mistakes in a real world case
-	Malin: ‘golden truth’ images are even pre-sorted before we classified anything by hand

## Image labeling  

How many images do we have and how many do we need?
-	We have 13 classes, and in it are maximal  ~1700/3 images and  ~1200/3 images on average (machine) labeled
-	Unlabeled images are 120 000 spears (so 3x120 000 images), plus further ~150 GB of images that are not uploaded yet
-	Ulf: You should work as if you always have 3 pictures on hand
-	We don’t have a ‘hard’ number of how many pictures we can label until a certain date
-	Our idea: when we have 20.000 images we can already start building networks and see how much is needed (as a minimum) for a good performing network

Classes vs features
-	Ulf: It is hard to tell what is better. It depends on the question what do I aim for. For a deep network you need labeled data, however, it won’t really get any better than what you gave ‘per hand’ into the machine (as a golden standard)
-	In machine learning you usually have data sets like MNIST or other sets where we have the same amount of pictures for every class. Here, we have another situation. MNIST could be good as a comparison data set (as it is also very homogeneous - different than your problem but might be similar). You are not working with 1000 of features but only a handful. You don’t label the classes but the features (which might be different).
-	Sophia: Yes, because this makes it much more flexible (to label features not classes); we might get the percentage of the classes from Silvan; also certain mistakes (when sorting wrong) are worse than others
-	Ulf: So, the central question isn’t so much the usual machine learning problem of mapping features to classes but extracting the features themselves
-	Richard: The idea is to be able to shift classes while keeping the features (for flexiility)
-	Sophia: Classes aren’t as hard bound (not even for Silvan), which makes it difficult (to only go with the classes)
-	If we should decide to go from features to classes and train with them, it wouldn’t be a lot of extra effort

Classical machine learning vs. (deep) neuronal networks
-	Ulf: Comparing the classic with the neuronal approach makes sense. For the neuronal, it’s more ‘let the data speak for itself’ while the classical is much more about preprocessing. Depeding on performance or usability, it makes sense to see which one rather to work with.
-	The other question here is, how good you can do the feature extraction. If this works well, it leaves the question what do you even want to classify with the neural network?
-	Richard: That’s what Mr. Hermeler already said. The feature extraction for asparagus is difficult and does not work properly or reliable enough for certain features/classes.
-	Ulf: As help for labeling, easy and critical cases should be identified, because rather the difficult cases than the obvious one’s are interesting to you
-	Sophia: Yes, that’s what we want to look at. Easily detectable features like length and width aren’t the problem, in contrast to a feature like ‘Blume’ or other grey zone cases.

Our focus beginning from now
-	We wish for a bit more guidance from our supervisors in situations when we have different approaches/ideas we could pursue. Guidance would help us a lot to decide which lead to follow.
-	Ulf: Concerning the momentarily situation, a big and labeled data set isn’t crucial. Knowing at which cases we want to look like and also how we want to look at them is the important part.
-	And yes, the 2nd step (from features to classes) is also important. But for now it might be better to see how the distribution is in general in the data set, to have representable sets, a representative sample (Stichprobe). We only need labeled data for testing and for training (independently from each other of course) and then it’s only a case of how much freedom you allow. It might already be at around 500 - 1000 images (per feature) until you can start with building a network.
-	You have the advantage that you have a big dataset. You might look at unsupervised learning or semi-supervised learning again.
-	Michael: So our goal is to have more than 10.000 images to go through?
-	Ulf: Every class should have approx. the same amount of images, or rather, every feature should have the same amount. If you know the general distribution of the classes, it might even be better to fit this number to the amount that you train/work with.
-	Another question is how gravely are certain mistakes. Don’t let images that are difficult to sort fall under the table but include them in the metric. 

## Kappa agreement  

Presentation of results
-	Malin: We wanted to try out the kappa agreement to compare the group’s performance for sorting
-	It looks like the kappa value is low, like we sort close to chance level, which we actually don’t do.
-	At the moment it looks like this measurement approach is not what we can use as a value for how well our group agrees on the image sorting.
-	Another value is the accuracy which is how often two people decided to sort an image differently. What isn’t included here is if the case is a critical one or not.
-	The accuracy is often above 90% for single features, worst are 82% for the feature ‘rust body’ and 81% for the feature ‘bended’.

Sorting categories binary vs. transitionally
-	What about introducing a scalar instead of binary decisions?
-	Sophia: The idea was to avoid more questions and also to reduce the bias that we might tend to avoid clear decision making when we can choose between ‘bended’ and ‘slightly bended’.

Should we use the kappa agreement, and, if yes, which value?
-	Malin: Would accuracy be a measure that we can use or should we go for another one?
-	Kappa agreement means a value between -1 to 1, where 0 would be chance level and 1 = 100% agreement.
-	Malin: In the internet we found different cases where people had problems with the kappa value. For example in one 2/3 of images were labeled correctly but the kappa value still said it is sorted by chance level. It is difficult to understand how the kappa value works but it seems one image will always be taken as the truth and then it is compared how well do other images fit to the first one. So if one person sorts ‘1 1 0 1’, the agreement compares what the other person sorted for at the place where the first said ‘0’. It seems relevant what value you (as first person) use first.
-	K(appa) = Pi (what we watched) – Pr (random value) /  1 – Pr 
-	Kappa was just one example we can use, we did not have a certain convition that made us use it. So we might look for another metric, or should we use the accuracy metric?
-	Ulf: Might be good to use other approaches. I don’t understand kappa enough to tell anything about it but it also seems to be a good measurement (if you understand how it works).
-	Something else could be the F1 score, which takes precision and recall into account and is for binary classification.
-	Ulf: You only use binary cases right now?
-	Malin: For the classifying right not we only used binary features except width and length (which are excluded from the kappa?).
-	Sophia: We have to be careful because we only sorted inside certain categories we even knew beforehand. For example, rust does not happen often in category 1A Anna and so forth.
-	Ulf: An advantage when using the same data set for more than two people, would be to see which cases are clear and which unclear to extract the interesting cases.
-	Ulf: It is interesting for the report how good the labels are, why it is like this, and how it does change between the person rating the spear –so how constant is our decision making.
-	What metric to use is also important but it is important to use numbers in general to solidify your results, and also to see what works well and what doesn’t.

## Next Steps  
  
Take home message  
-	Ulf: Look at your data set: How similar is it to other data sets, to filter out priorities/biases for certain mistakes. The data consists of very homogenous images, so what are the consequences for your results?
-	Put this all together and present that to get an overview (for you and also for us).
   
What still to discuss
- Manager roles
- 101 hand labeling: Go through images with the group for learning how to sort the spears correctly (= get a better agreement)


