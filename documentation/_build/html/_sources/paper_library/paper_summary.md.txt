# Asparagus classification


## Automated design of a computer vision system for visual food quality evaluation

| title          | Automated Design of a Computer Vision System for Visual Food Quality Evaluation         |
|----------------|-----------------------------------------------------------------------------------------|
| helpful        | 3/5                                                                                     |
| authors        | @Mery2013                                                                               |
| year           | 2013                                                                                    |
| summary        | proposes a general framework that designs a computer vision system automatically, i.e., it finds —without human interaction— the features and the classifiers for a given application avoiding the classical trial and error framework commonly used by human designers. The key idea of the proposed framework is to select *automatically* from a large set of features and a bank of classifiers those features and classifiers that achieve the highest performance. |
| images         | optical and/or X-ray images; 8 different sets of food images (blueberries, potatoes etc.) with varying pixel size
| labeling       | images have to be labelled manually by experts to establish class for training
| framework      | 1) data collection, 2) large feature extraction, 3) feature selection, 4) classifier selection
| features       | more than 3,000 features containing geometric (location, size, shape etc.) and intensity (RGB, L* a* b* etc.) information |
| feature algorithms | sequential forward selection (SFS), ranking by class separability criteria (rank)
| classifiers    | best performance by: Mahalanobis distance, Linear discriminant analysis (LDA), Nearest neighbors (k-NN), Support vector machines (SVM), Neural networks (NN)
| implementation | Balu Matlab Toolbox
| performance    | tested framework on eight different food quality evaluation problems yielding a classification performance of 95 % or more in every case



## Classification of asparagus sections

| title        | Classification of processing asparagus sections using color images |
|--------------|-------------------------------------------------------------------------------------------------|
| helpful            | 2/5                                                                                     |
| authors            | @donis2016classification                                                                |
| year               | 2016                                                                                    |
| summary            | sort (green) asparagus into three classes: tips, mid-stem pieces and bottom-stem pieces |
| images             | 955 color images                                                                        |
| data acquisition   | manually cleaned with water, each fresh asparagus cut into 50 mm sections, scanned using a 48 bit color scanner, resolution of 816 x 1123 pixels  |
| features           | extraction of a total of 1931 color, textural, and geometric features                   |
| approach           | 43 features were found to be effective in designing a neural-network classifier         |
| labeling           | images were labeled by the research team                                                |
| image segmentation | cropped using Matlab R2012a, segment the asparagus from its background                  |
| feature selection  | 10 intensity images were obtained from each asparagus, extracted features included standard features, invariant shape moments, Haralick textural features local binary patterns, and Gabor filters                                                                       |
| performance        | 4-fold cross-validated overall performance accuracy of 90.2% (±2.2%)                    |



## Comparison of three algorithms for the classification of table olives

| title        | Comparison of three algorithms in the classification of table olives by means of computer vision|
|--------------|-------------------------------------------------------------------------------------------------|
| helpful      | 2/5                                                                                             |
| authors      | @diaz2004comparison                                                                             |
| year         | 2003                                                                                            |
| summary      | classification of table olive into 4 different quality categories: In this study, olives have been characterised identifying the most common defects and its colorimetric properties. Then, a capture and colour images processing system was used to extract the information from each olive. Afterwards, a grading system which is able to learn from the characteristic parameters of the olives previously classified by experts, was used to determine to which category each olive was more likely to belong. Finally, three different learning and classification algorithms have been applied to extracted parameters, comparing and analysing the grading results. |
| data         | 400 samples (approximately half were used for training and half to validate)                    |
| labeling     | olives, previously measured according to their size, were labeled by experts                    |
| illumination | The olives are in brine, and its sparkle can mask the colorimetric information about the defects. In order to minimise this effect it was decided to place high frequency fluorescent tubes around the pitch area.                                                               |
| images       | colorimetric analysis. Every image captured has a resolution of 768x576 pixels, containing 6 rows with 11 olives |
| features     | The parameters obtained from each olive in each image are the pixel number of lighter skin (Skin 1), of darker skin or olive profile (Skin 2), of light defect (Stain 1), of dark defect as a bite (Stain 2) and of unusual dark colour (Stain 3). Taking three images of each olive, a total number of 15 parameters can be obtained. |
| 1. algorithm | classical Bayesian discriminant analysis (simplification: Mahalanobis distance version to the $1-k$ nearest neighbours was used) |
| 2. algorithm | PLS-DA (partial least squares) multivariant discriminant analysis                                                                |
| 3. algorithm | neural networks with Rprop. The input layer is made up of 15 neurons, one for each parameter. The hidden layer is also made up of 15 neurons, connected to all the input neurons and output neurons. And finally, there are 4 neurons in the output layer, one for each olive category.                              |
| performance    | Neural network based on resilient back-propagation is the algorithm with the best results in discriminating the forth classes  |



## Classification of beans

| title          | A classification system for beans using computer vision system and artificial neural networks         |
|----------------|-----------------------------------------------------------------------------------------|
| helpful        | 1/5                                                                                                   |
| authors        | @KILIC2007897                                                                                         |
| year           | 2013                                                                                                  |
| summary        |  A computer vision system (CVS) was developed for the quality inspection of beans, based on size and color quantification of samples. The system consisted of a hardware (=developed to capture a standard image from the samples) and a software (=coded in Matlab for segmentation, morphological operation and color quantification of the samples).                                                                                                                |
| data           | 511 bean samples; 69 samples used for training, 71 for validation, and 371 for testing of the network |
| image          | ~ 1280 x 960 pixels, rows of 62 beans per picture with black background                               |
| labeling       | manually                                                                                              |
| features       | length, width, color distribution (in RGB)                                                            |
| implementation | Matlab Artificial Neural Networks Toolbox with 12 neurons in input and hidden layer and 2 in output layer |
| performance    | The automated system was able to correctly classify 99.3% of white beans, 93.3% of yellow–green damaged beans, 69.1% of black damaged beans, 74.5% of low damaged beans and 93.8% of highly damaged beans; overall correct classification rate obtained was 90.6%                          |



## Classification of potatoes 

| title        | Grading of potatoes                                                                     |
|--------------|-----------------------------------------------------------------------------------------|
| helpful      | 1/5                                                                                     |
| authors      | @PEDRESCHI2016369                                                                       |
| year         | 2016                                                                                    |
| summary      | describes how potatoes are graded automatically and which are the principal potato features and surface defects that determine the strategies one has to apply for a proper grading |
| illumination | @leon2006color recommend a controlled illumination setup and present a computational solution that allows the obtaining of digital images in L\*a\*b\* color units for each pixel of the digital RGB image. |
| features     | major kinds: color, size, shape, and texture. The proper combination of different kinds of image features, such as for instance combining size with shape and color with texture, can normally increase the accuracy of the results. Sometimes such a combination might even reveal some quality attributes that cannot be identified by using only a single kind of image feature. |
| color segmentation | the majority of external defects and diseases are identified by its color, which makes the classification of pixels into homogeneous regions an important part of the algorithm. Multilayer feed-forward neural networks (MLFN-NN) and statistical discriminate functions have been used successfully for the segmentation of potato images.|
| shape classification| Fourier Descriptors (FD) and linear discriminant analysis (LDA) are used to discriminate between good and misshapen potatoes. A single shape model is not enough to segment all potato cultivars into good and misshapen classes. Good-shaped potatoes may vary from round, oval, and extreme oval. Therefore different shape models are created for different potato cultivars. A shape training set and shape test were created for each cultivar to discriminate between good potatoes and misshapen potatoes (@jacco2000high)|
|@elmasry2012line | developed a fast and accurate computer-based machine vision system for detecting irregular potatoes in real time. A database of images from potatoes with different shapes and sizes was formulated. Some essential geometrical features such as perimeter, centroid, area, moment of inertia, length, and width were extracted from each image. Eight shape parameters originated from size features and Fourier transform were calculated for each image in the database. All extracted shape parameters were entered in a stepwise linear discriminant analysis to extract the most important parameters that most characterized the regularity of potatoes. Based on stepwise linear discriminant analysis, two shape features (roundness and extent) and four Fourier-shape descriptors were found to be effective in sorting regular and irregular potatoes.|



## Comparing  Deep Learning And Support Vector Machines for Autonomous Waste Sorting

| title          | Comparing  Deep Learning And Support Vector Machines for Autonomous Waste Sorting       |
|----------------|-----------------------------------------------------------------------------------------|
| helpful        | 2/5                                                                                     |
| authors        | @7777453                                                                                |
| year           | 2016                                                                                    |
| summary        | comparing SVM and Deep Learning (with CNN) for autonomous waste sorting , provided only by images of waste. The needed to be categorized into three different groups (by texture: plastic, paper, metal), using raspberry pi 3. Approach with the best accuracy (SVM: 94.8%, CNN: 83%) and classification speed was implemented. CNN was worse due to image squashing (limitation in memory size for the GPU) |
| images         | total of 2000 pictures (taken by pi camera), 1/3 for each wastegroup (60% Training, 20% Validation, 20% testing), training images were augmented artificially to 6000 images.
| labeling       | images have to be labelled manually by experts to establish class for training
| feature algorithms | >bag of features< for SVM trained through >bagOfFeatures< function from Matlab
| classifiers    | best performance by: Support vector machines (SVM), CNN suffered from overfitting due to the small amount of pictures and the squashing.
| performance    | SVM: 94.8%, CNN: 83%


## Paper for Kappa agreement

| title          | Interrater reliability: the kappa statistic       |
|----------------|-----------------------------------------------------------------------------------------|
| helpful        | 4/5                                                                                     |
| authors        | @PMC3900052                                                                            |
| year           | 2012                                                                                    |
| summary        | This paper includes information about the kappa agreement. The kappa can range from −1 to +1. Cohen suggests the Kappa result to be interpreted as follows: values ≤ 0 as indicating no agreement and 0.01–0.20 as none to slight, 0.21–0.40 as fair, 0.41– 0.60 as moderate, 0.61–0.80 as substantial, and 0.81–1.00 as almost perfect agreement.  |
