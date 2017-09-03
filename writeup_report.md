**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-hog_image.png
[image2]: ./output_images/car-bin_spatial.png
[image3]: ./output_images/car-color_hist.png
[image4]: ./output_images/image_1.png
[image5]: ./output_images/image_2.png
[image6]: ./output_images/image_3.png
[image7]: ./output_images/image_4.png
[image8]: ./output_images/image_5.png
[image9]: ./output_images/image_6.png
[image10]: ./output_images/multi_scale_1.png
[image11]: ./output_images/multi_scale_2.png
[image12]: ./output_images/multi_scale_3.png
[image13]: ./output_images/final.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is in `get_hog_features()` in `features_extract.py`. I've also used color histograms and spatial binning. The code is in `color_hist()` and `bin_spatial()`, respectively.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` on the first channel of the image:

![alt text][image1]

The same example image that has been resized to 16x16 by function `bin_spatial()`:

![alt text][image2]

Below are the color histogram for the same example image with 8 bins:

![alt text][image3]

Note that although we know that `matplotlib.image` reads PNG images in a scale of 0 to 1, for some reason, `cv2.cvtColor()` still scales the image from 0 to 1, not from 0 to 256 as other color spaces do. To avoid this issue, I read PNG images using `cv2.imread()` so the image scaled from 0 to 256 and are in BGR format. As the input images from the video frames are in RGB format, function `convert_color()` in `features_extract.py` is implemented to handle different formats.

####2. Explain how you settled on your final choice of HOG parameters and how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn` package. `StandardScaler().fit()` has been used to normalize the feature vectors and `train_test_split()` to split data into 80% for training and 20% for testing. The code is in `classifier.py` between line 74 to line 95.

I first started from the same the parameters as on the class slides. And then randomly increased or decreased the values to see whether the accuracy on the testing data improves. Below are my final parameters and the accuracy is 0.9935.

```python
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial = 16 # number of spatial bins
histbin = 8 # number of histogram bins
orient = 8 # numbre of HOG orientations bins (typically between 6 and 12)
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or 'ALL'
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The window sliding search function is in `find_cars()` in `detection.py`. Most of the code follows the class slides. First, an image is resized based on the given scaling parameter, and hog features are extracted for the region of the interest in the scaled image. Then as line 30-38, the number of steps in units of `cells_per_step` is calculated, and then a step-by-step window search is performed. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I combined hog features, color histograms, and spatial binning. Their parameters are as shown above. A window searching with scale of 1.25 and `cells_per_step` of 2 is performed on image for position between `y = 400` to `y = 650`. The code is in function `run_images()` in `detection.py`.

Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Video implementation is in function `process_image_with_vehicle_detection()` in `detection.py`.

To filter false positives, heatmap is created based on positive detections in each frame of the video, and then the heatmap is thresholded to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

In my implementation, I sum the heatmaps of the last 5 frames and the heatmap of the current frame, and thresholded by `10`.   

In addition, to increase the positive detections, a multi-scale window search was performed here. Scale of 1.25 and 1.5 are performed on the image at position between `y = 400` to `y = 650`. Moreover, such scaling doesn't give enough positive detections on vehicles at right-upper region, which are the most further away from the camera (i.e. the vehicles are much smaller). The white car in the below image is an example:

![alt text][image10]

Meanwhile, scaling of 1.25 and 1.5 also failed to detect vehicles at the very right-bottom of the image, in particular, when the car is exiting from the video. Please see the white car at this image as an example:

![alt text][image11]

Based on these regions, some special scalings have performed at positions between `x = 900`, `x = 1100` and `x = 1280. Code is at line 169-175 in function `process_image_with_vehicle_detection()` in `detection.py`:

```python
for ystart, ystop, xstart, xstop, scale, step in [(y1, y4, 0, 1280,
    1.25, 2), (y1, y4, 0, 1280, 1.5, 2), (y1, y4, 1100, 1280, 1.25, 1),
    (y1, y4, 900, 1100, 0.75, 2), (y1, y4, 1100, 1280, 1.5, 1)]:
```

where `y1 = 400` and `y4 = 650`. Here is an example showing different regions:

![alt text][image12]

The scaling idea is based on the relative vehicle size to decide scale up/down. For example, the furthest vehicle should be scaled up (thus, set to 0.75), and to help detect the exiting vehicle, `cells_per_step = 1` is given using the existing scale of 1.25 and 1.5 to increase chances of detection.

Note that the region is hard coded which can only work on the project video. A better approach is needed.

Here's an example result showing the resulting bounding boxes overlaid on the last frame after six frames of video:

![alt text][image13]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Before I caught the issue of `YCrCb` 0-1 scale, the classifier gave 0.98 accuracy, and somehow produced false positives that were not filtered out heatmap thresholding, and I had to play with `svc.decision_function()` instead of `svc.predict()`.

The other thing I could do better is the one mentioned above, the hard-coded region, that can only works on the project video. I tried to use the lane detected from the previous project to estimate the region, but didn't get reliable results within limited time I have now.

The last thing is that even I've spent many efforts on those regions for the vehicles harder to detect, in particular, the vehicles exiting the video, the results are still not that good. Perhaps the training image of such vehicles should be added. Most of images we use were taken from the back of the vehicles.
