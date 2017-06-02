
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  

# ## Import Packages

# In[324]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')


# ## Read in an Image

# In[325]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[326]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=30):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    (h, w, color_dim) = image.shape

    left_results = [] # instead of lines, just store the center point for each line
    right_results = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            # perhaps using center point can further reduce noise?
            #line_center_point = np.array([x1 + (x2-x1)/2, y1+ (y2-y1)/2])
            if (x2-x1) != 0: 
                derivative = (y2-y1)/(x2-x1)
                b = y1-derivative*x1
                if derivative < -0.40: # to reduce noise
                    left_results.append((derivative, b) )
                elif derivative >= 0.50: # to reduce noise
                    right_results.append((derivative, b) )
    
    if len(left_results) > 0:
        left_results = np.array(left_results)
        (average_derivative,average_b) = np.mean(left_results, axis=0)
        #print("LEFT", average_derivative, average_b)
        stop_y = int(h//2+h//12) # UPPER LEFT Y
        # y=ax-b => for unknown x = (y-b)/a
        stop_x = int((stop_y-average_b)/average_derivative)
        start_y = int(h)
        start_x = int((start_y-average_b)/average_derivative)
        #print("startx")
        #print(start_x)
        #print(stop_x)
        #print(start_y)
        #print(stop_y)
        cv2.line(img, (start_x, start_y), (stop_x, stop_y), color, thickness)

    if len(right_results) > 0:
        right_results = np.array(right_results)
        (average_derivative,average_b) = np.mean(right_results, axis=0)
        #print("RIGHT", average_derivative, average_b)
        stop_y = int(h//2+h//12) # UPPER RIGHT Y (SAME AS FOR LEFT)
         # y=ax-b => for unknown x = (y-b)/a
        stop_x = int((stop_y-average_b)/average_derivative)
        start_y = int(h)
        start_x = int((start_y-average_b)/average_derivative)
        cv2.line(img, (start_x,start_y), (stop_x,stop_y), color, thickness)
        


        

        
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return (line_img, lines) # also returning lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

#### EXTRA FUNCTIONS BY AMUND

def build_vertices():
    # 1. EXTRACT REGION OF INTEREST
    vertices = np.array([[(w//15,h),  # LOWER LEFT
                    (w//2-w//50, h//2+h//15), # UPPER LEFT
                    (w//2+w//50, h//2+h//15), # UPPER RIGHT
                    (w-w//15,h)]],  # LOWER RIGHT
                    dtype=np.int32)
    return vertices

def create_center_points_from_hough_lines(lines):
    i = 0
    left_results = [] # instead of lines, just store the center point for each line
    right_results = [] # instead of lines, just store the center point for each line.
    for line in lines:
        for (x1,y1,x2,y2) in line:
            # note: x increases towards the right, but y increases downwards
            line_center_point = np.array([x1 + (x2-x1)/2, y1+ (y2-y1)/2])
            if (x2-x1) != 0: 
                derivative = (y2-y1)/(x2-x1)
                b = y1-derivative*x1
                if derivative < -0.40: # only keep if in the neighbourhood of ROUGH LEFT lane derivative
                    left_results.append((derivative, b) )
                elif derivative >= 0.40: # similar, only keep if in the neighbourhood of ROUGH RIGHT lane derivative
                    right_results.append((derivative, b) )
    return (left_results, right_results)
            
                        
def calculate_average_and_median_derivative_between_center_points(center_points):
    # calculate average derivative between center points 
    # note: double counting here.. 
    ders = []
    for i, outer_center_point in enumerate(center_points):
        for j, inner_center_point in enumerate(center_points):
            if i == j:
                continue
            x_delta = outer_center_point[0]-inner_center_point[0]
            if x_delta != 0:
                y_delta = outer_center_point[1]-inner_center_point[1]
                # perhaps check for negative
                derivative = y_delta / x_delta
                if derivative < 0:
                    derivative = -derivative
                ders.append(derivative)
    average_derivative = np.average(ders)
    median_derivative = np.median(ders)
    return (average_derivative, median_derivative)
            
def plot_histogram(points, figure_number=0):
    plt.figure(figure_number)
    import matplotlib.mlab as mlab
    # the histogram of the data
    # s/derivates/x
    n, bins, patches = plt.hist(points, 50, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)

    plt.xlabel('Derivative')
    plt.ylabel('Frequency')
    plt.title(r'$\mathrm{Histogram\ of\ Hough Line Derivates:}$')
    plt.axis([0, 1.0, 0, 2])
    plt.grid(True)
    
def plot_image_with_figure_number(image, figure_number=0, cmap=None):
    plt.figure(figure_number)
    if cmap is None:
        plt.imshow(image)    
    else:
        plt.imshow(image,cmap=cmap)
        


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[327]:


import os
os.listdir("test_images/")


# In[328]:


#
###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
#red_threshold = 200
#green_threshold = 200
#blue_threshold = 200
######

#rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Do a boolean or with the "|" character to identify
# pixels below the thresholds
#thresholds = (image[:,:,0] < rgb_threshold[0]) \
#            | (image[:,:,1] < rgb_threshold[1]) \
#            | (image[:,:,2] < rgb_threshold[2])
#color_select[thresholds] = [0,0,0]




# https://extr3metech.wordpress.com/2012/09/23/convert-photo-to-grayscale-with-python-opencv/
# https://stackoverflow.com/questions/38877102/how-to-detect-red-color-in-opencv-python
# red = numpy.uint8([[[0,0,255]]])



# 1) convert image to grayscale
# 2) do color selection - keep white (and later yellow lines) - to remove most of stuff
#   (perhaps make the remaining colors transparent?)
# 3) select color region - only containing lanes
# 4) detect edges with canny edge detection
# 5) do hough transform and find lines
# 6) use line information to draw on top of lanes

# Display the image                 
#plt.imshow(image)
#plt.imshow(grayscale_image, cmap='gray')
#plt.imshow(mask, cmap=')



# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[329]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.



def LaneFindingPipeline(image):
    # 1. REGION OF INTEREST
    (h, w, color_dim) = image.shape
    vertices = build_vertices()
    region_of_interest_image = region_of_interest(image, vertices)

    # 2. GRAYSCALE
    grayscale_image = grayscale(region_of_interest_image)
    #print(grayscale_image[0][0])

    # 3. EXTRACT ONLY LIKELY LANES (CLOSE TO WHITE)
    # only 1 dimension of color in grayscale, so simpler thresholds
    grayscale_select_image = np.copy(grayscale_image)
    white_threshold = (grayscale_image[:,:] < 220)
    grayscale_select_image[white_threshold] = 0
    # FOR YELLOW - can potentially overlay two threshold solutions?
    
    # rgb = 255,255,0 is yellow.
    yellow_threshold = [220,220,30]
    yellow_select_image = np.copy(region_of_interest_image)
    yellow_thresholds = (image[:,:,0] > yellow_threshold[0])             | (image[:,:,1] > yellow_threshold[1])             | (image[:,:,2] < yellow_threshold[2])
            
    yellow_select_image[yellow_thresholds] = [0,0,0]
    grayscale_from_yellow = grayscale(yellow_select_image)
    
    # combine yellow and gray
    combined_gray_image = np.maximum(grayscale_select_image, grayscale_from_yellow)


    # 4. BLURRING + CANNY EDGE DETECTION
    blur_kernel_size = 3 # 5, 7, ..
    blur_gray = cv2.GaussianBlur(combined_gray_image,(blur_kernel_size, blur_kernel_size),0)
    #canny_edge_image = canny(grayscale_select_image, low_threshold=10, high_threshold=150)
    canny_edge_image = canny(blur_gray, low_threshold=10, high_threshold=150)

    # 5. HOUGH
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 3 #minimum number of pixels making up a line
    max_line_gap = 1    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    hough_image, lines = hough_lines(canny_edge_image, rho, theta, threshold, min_line_len, max_line_gap)
    
    # 7. WEIGHTED IMAGE - combine original image with the ones with lane markings on
    weighted_image = weighted_img(hough_image, image, α=0.8, β=1., λ=0.)
    
    
    
    
    # 8. DEBUG PLOTTING OF IMAGES GENERATED THROUGH PIPELINE STEPS
    #figure_number = 0
    #plot_image_with_figure_number(image, figure_number)
    #figure_number += 1

    #plot_image_with_figure_number(region_of_interest_image, figure_number)
    #figure_number += 1

    #plot_image_with_figure_number(grayscale_select_image, figure_number, cmap='gray')
    #figure_number += 1

    #plot_image_with_figure_number(canny_edge_image, figure_number, cmap='gray')
    #figure_number += 1

    #plot_image_with_figure_number(hough_image, figure_number, cmap='gray')
    #figure_number += 1

    #plot_image_with_figure_number(weighted_image, figure_number) #, cmap='gray')
    #figure_number += 1
    #plt.show()
    
    #plot_image_with_figure_number(yellow_select_image, figure_number)
    #figure_number += 1
    #plt.show()
    
    #plot_image_with_figure_number(combined_gray_image, figure_number, cmap='gray')
    #figure_number += 1
    #plt.show()
    

    return weighted_image

#print(lines) - e.g.
# [[[758 480 782 494]],  [[754 486 786 508]], ..

# TRIED, but failed perhaps can use the region of interest coordinates to give a rough estimate of
# what the derivatives for the lane would be, and then only keep the ones in the
# neighbourhood of those to reduce noise

# PARTIALLY TESTED - NEW STRATEGY
# instead of using hough lines directly, just calculate center point for each hough line
# and find average derivative between the center points instead - since it is likely that the small 
# hough lines may point in many different directions, but the center point of each line is likely to
# be more representative. Can then calculate derivatives between all center points, and perhaps
# choose the median. Can then use that median to find a good b-value, ref: y = ax-b

# calculate average derivative between center points 
# note: double counting here..

### HISTOGRAM FOR HOUGH LINE DERIVATIVES
 
#plot_histogram(left_center_points, figure_number)
#figure_number += 1
#plot_histogram(left)


# 6. SELECT LANES 

# 0) READ IMAGE
image = mpimg.imread('test_images/solidWhiteRight.jpg')
weighted_image = LaneFindingPipeline(image)

figure_number = 1
plot_image_with_figure_number(weighted_image, figure_number) #, cmap='gray')
figure_number += 1
plt.show()


# In[331]:


# TEST ALL IMAGES

figure_number = 1

for image_name in os.listdir("test_images/"):
    full_path = "test_images/" + image_name
    image =  mpimg.imread(full_path)
    weighted_image = LaneFindingPipeline(image)
    plot_image_with_figure_number(weighted_image, figure_number) #, cmap='gray')
    figure_number += 1
plt.show()


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**

# In[335]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[338]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = LaneFindingPipeline(image)
    return result


# Let's try the one with the solid white lane on the right first ...

# In[339]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[340]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[341]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[342]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[343]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

