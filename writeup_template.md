# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of the following steps: 
1) First I selected the region of interest (with hand-made vertices), 
2) Converted the image to grayscale
3) Extracted likely white lane information from the grayscale image
4) Extracted likely yellow lane information from the (colorized) region of interest image
5) Converted the yellow lane information image to grayscale
6) Combined the likely yellow and white lane grayscale images into a new grayscale image (using max value)
7) Did a gaussian blur followed by canny edge detection
8) Did a hough image creation, I also modified the draw_lines function by calculating average derivative and b value (i.e. calculating y = x-b for all the hough lines to find a and b, and then average over them).
    (side note: believe it perhaps could have been smarter to use hough line center points instead of hough lines, since
    the directions of them seem sometimes a bit unstable, and then use average of derivatives between center points instead)
9) Used the weighted image to overlay the hough image with lane detection on top of the original image

### 2. Identify potential shortcomings with your current pipeline

For some reason it didn't work well on the last image (as you can see in the notebook), perhaps due to that hough lines have some random directinos that create distortions - perhaps the center line approach could work better. 

### 3. Suggest possible improvements to your pipeline

Use the center of hough line approach, and calculate derivatives between them - believe this could smooth out some randomness in the direction of hough lines.
