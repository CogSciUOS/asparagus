# Friday, June 7, 2019 (Documentation, Farm Visit, Progress Feature Extraction)

Ulf, Axel, Katha, Maren, Sophia, Luana, Thomas, Richard, Subir

## Summary of Richard's visit to the other farm 
- he collected images and suggests going there another time to collect even more, as well as gather data for evaluation by using the already classified asparagus
- the pictures from the other machine are better quality, both in terms of lighting and in terms of resolution, especially visible in the detailed pictures of heads
- it is very important to clean the camera before using the machine (can be done with a soft/microfiber cloth)
- figured out why the images couldn't be saved so now that issue can be fixed
- the machine there cannot be connected to wifi, even though the other farm does has internet e.g. cameras are connected via LAN. It seems there haven’t been any attempts to fix that (from anyone there)

## Updates from the meeting
- Subir and Malin went to Rheine on Monday to collect images
- we have a Google doc now where everybody can collaborate and record what they did, what methods they used, what worked and didn’t work, what they are working on. There is also an initial structure for the final report where we can already start writing bullet points about the information that will be included in the end
- Katha presented the way we will maintain the documentation for our code. We will be using Sphinx and readthedocs for hosting the code. So far she is in charge of this task, but she wouldn't mind having someone join her on it
- Luana will do a presentation on data augmentation/working with small data sets in 2 weeks (21.06)
- Sophia goes to Rheine this weekend (7-8.06) to collect labelled data and would like for someone to go with her

## Progress on feature extraction tasks
- length and curvature extraction have been implemented and work relatively well
- rust/violet extraction is difficult without labelled data (or without a reference image), so this feature is on hold until that is available
- the rest of the tasks should be discussed next time

## For next time:
- most important point of discussion should be assigning clear task to everyone, with a due date. This is to ensure that nobody is left out and everyone has a clear view of their contribution
- another priority is collecting as many images as possible, so we should create a schedule for everyone to go to Rheine at least once and do this