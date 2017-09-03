import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import collections
from image_processor import ImageProcessor, plotTwo

# Global parameters
ym_per_pixel = 30/720 # meters per pixel in y dimension
xm_per_pixel = 3.7/700 # meters per pixel in x dimension
ploty = np.linspace(0, 719, 720) # y range of image
num_failed_allowed = 10 # max number of consecutive failed frames
num_recent_fits = 10 # max number of the past fits to average

# Use window slding to detect lines
def run_one_image(fname, plot = False, debug = False):
    img = mpimg.imread(fname)
    name = ' (' + os.path.splitext(os.path.basename(fname))[0] + ')'
    # Pre-process all the images: distort, thresholding, and perspective transform
    undist = processor.undistort(img)
    plotTwo((img, undist), ('Original'+name, 'Undistorted'+name))

    thresh = processor.thresholding(undist, debug)
    plotTwo((undist, thresh), ('Undistorted'+name, 'Thresholding'+name))

    # For debugging, warp on the undistort image.
    if debug:
        processor.perspective_transform(undist, debug)
    warped = processor.perspective_transform(thresh)
    plotTwo((thresh, warped), ('Thresholding'+name, 'Warped'+name))

    # Use window sliding to find the lines
    left_fit, right_fit = find_lane_lines_sliding_window(warped, plot)

    if debug:
        print(name, ": ", left_fit, ", ", right_fit)
        
    result = draw_lane(img, warped, left_fit, right_fit)
    
    if plot:
        plt.title('Detected lane'+name)
        plt.imshow(result)
        plt.show(block=True)


    # Calculate radius_of_curvature
    curv = (cal_radius_of_curvature(left_fit) + cal_radius_of_curvature(right_fit))/2
    draw_radius_of_curvature(result, curv)
    draw_vehicle_position(result, left_fit, right_fit)

    if debug:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        diff = right_fitx - left_fitx
        min_width = np.min(diff)
        max_width = np.max(diff)
        cv2.putText(result, 'min_width = ' + str(round(min_width, 3))+' (pixel)',(50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(result, 'max_width = ' + str(round(max_width, 3))+' (pixel)',(50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        print("min_width: ", min_width)
        print("max_width: ", max_width)

    plt.title('Detected lane with info'+name)
    plt.imshow(result)
    plt.show(block=True)


def run_all_test_images(imageDir, plot = False, debug = False):
    testFiles = glob.glob(imageDir+'/*.jpg')
    for fname in testFiles:
        run_one_image(fname, plot, debug)

def find_lane_lines_sliding_window(img, plot=False):
    # Parameters setting
    num_windows = 9
    window_height = np.int(img.shape[0]/num_windows)
    margin = 80 # widnow width = 2 * margin
    min_pixels = 50 # minimum number of pixels found in one window

    # Create an output image to draw on and  visualize the result
    if plot == True:
        out_img = np.dstack((img, img, img))*255

    # Starting points: The peaks of the histogram at the left half and the right half.
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint

    # X and y positions where the pixels are nonzero
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # For pixel found within windows, record the indices to nonzero array
    left_lane_indices = []
    right_lane_indices = []

    for window in range(num_windows):
        # Window boundaries
        win_y = (img.shape[0] - (window+1)*window_height, img.shape[0] - window*window_height)
        win_left_x = (leftx_current - margin, leftx_current + margin)
        win_right_x = (rightx_current - margin, rightx_current + margin)

        if plot == True:
            # Draw the windows
            cv2.rectangle(out_img, (win_left_x[0], win_y[0]), (win_left_x[1], win_y[1]), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_right_x[0], win_y[0]), (win_right_x[1], win_y[1]), (0, 255, 0), 2)

        # Find nonzero pixels within the window
        nonzero_left_indices = ((nonzeroy >= win_y[0]) & (nonzeroy < win_y[1]) & (nonzerox >= win_left_x[0]) & (nonzerox < win_left_x[1])).nonzero()[0]
        nonzero_right_indices = ((nonzeroy >= win_y[0]) & (nonzeroy < win_y[1]) & (nonzerox >= win_right_x[0]) & (nonzerox < win_right_x[1])).nonzero()[0]
        # Record the indices of nonzero pixels
        left_lane_indices.append(nonzero_left_indices)
        right_lane_indices.append(nonzero_right_indices)
        # Move to the next widnow if > minimum pixels found
        if len(nonzero_left_indices) > min_pixels:
            leftx_current = np.int(np.mean(nonzerox[nonzero_left_indices]))
        if len(nonzero_right_indices) > min_pixels:
            rightx_current = np.int(np.mean(nonzerox[nonzero_right_indices]))
    
    # Concatenate the indices array
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)
    # Fit a second order polynomial
    leftx = nonzerox[left_lane_indices]
    lefty = nonzeroy[left_lane_indices]
    rightx = nonzerox[right_lane_indices]
    righty = nonzeroy[right_lane_indices]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Plot the current fitted polynomial
    if plot == True:
        # Mark the pixels in the window: left with red, right with blue
        out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [0, 0, 255]
        plt.imshow(out_img)

        # Plot the fitted polynomial
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)

        plt.show(block=True)

    return left_fit, right_fit

def find_lane_lines_using_previous_fit(img, left_fit, right_fit, plot=False):
    # X and y positions where the pixels are nonzero
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100 # widnow width = 2 * margin
    # Indices within the margin of the polynomial
    left_lane_indices = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]) - margin) & 
                         (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2]) + margin))
    right_lane_indices = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]) - margin) & 
                          (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2]) + margin))
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_indices]
    lefty = nonzeroy[left_lane_indices]
    rightx = nonzerox[right_lane_indices]
    righty = nonzeroy[right_lane_indices]
    # Fit a second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if plot == True:
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
        # Mark the pixels in the window: left with red, right with blue
        out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [0, 0, 255]
        plt.imshow(out_img)

	# Generate a polygon to illustrate the search window area
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # window img
        window_img = np.zeros_like(out_img)
        # Draw the lane onto the window image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)

        plt.show(block=True)

    return left_fit, right_fit

def draw_lane(img, warped, left_fit, right_fit):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    if img.shape[2] == 3:
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    else:
        # Images dumped from video have 4 channels.
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero, warp_zero))

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, processor.invM, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, new_warp, 0.3, 0)

    return result

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # x values of the last n fits of the line
        self.recent_xfitted = collections.deque(maxlen=num_recent_fits)
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # radius of curvature of the best fit
        self.radius_of_curvature = None 
        # frame number in the video
        self.frame_counter = -1
        # number of consecutive failed frames 
        self.num_failed = 0

    def add_fit(self, fit, valid_line):
        self.frame_counter += 1

        if not valid_line:
            self.num_failed += 1
        else:
            self.current_fit = fit

            fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
            self.recent_xfitted.append(fitx)

            avg_fitx = np.average(self.recent_xfitted, axis=0)
            self.best_fit = np.polyfit(ploty, avg_fitx, 2)
            self.radius_of_curvature = cal_radius_of_curvature(self.best_fit)

    def do_slide_window(self):
        if self.best_fit == None:
            return True # Start from scratch
        if self.num_failed >= num_failed_allowed:
            self.reset()
            return True
        return False

    def reset(self):
        self.current_fit = None
        self.best_fit = None
        self.radius_of_curvature = None
        self.num_of_faileds = 0

    def check_line(self, fit):
        # Check radius_ofcurvature change
        curverad = cal_radius_of_curvature(fit)
        change = abs(curverad - self.radius_of_curvature)/self.radius_of_curvature
        if change > 0.3:
            return False

        # Check diff of polynomial cofficients between \fit and the best fit
        diff = abs(fit - self.best_fit)
        if diff[0] > 0.001 or diff[1] > 1 or diff[2] > 100:
            return False

        return True

    # The point of the line at the bottom image:
    def bottom(self, fit):
        pt = self.best_fit[0]*ploty[-1]**2 + self.best_fit[1]*ploty[-1] + self.best_fit[2]
        return pt

    # In real world pixel
    def bottom_pixel(self):
        return self.bottom() * xm_per_pixel

def cal_radius_of_curvature(fit):
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    # Fit new polynomials to x, y in world space
    fit_cr = np.polyfit(ploty * ym_per_pixel, fitx * xm_per_pixel, 2)
    y_eval = np.max(ploty)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pixel + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

# Draw radius of curvature on the result image
def draw_radius_of_curvature(img, curv):
    cv2.putText(img, 'Radius of curvature = ' + str(round(curv, 3))+' m',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

# Draw vehicle position from the center of the lane
def draw_vehicle_position(img, left_fit, right_fit):
    left_bottom = left_fit[0]*ploty[-1]**2 + left_fit[1]*ploty[-1] + left_fit[2]
    right_bottom = right_fit[0]*ploty[-1]**2 + right_fit[1]*ploty[-1] + right_fit[2]
    lane_center = (left_bottom + right_bottom) / 2
    vehicle_pos = img.shape[1]/2
    diff = (vehicle_pos - lane_center) * xm_per_pixel
    l_or_r = ' left' if diff < 0 else ' right'
    
    cv2.putText(img, 'Vehicle position : ' + str(abs(round(diff, 3)))+' m'+l_or_r+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

def process_image(img, plot = False, dump = False, to_draw_img = None):
    undist = processor.undistort(img)
    thresh = processor.thresholding(undist)
    warped = processor.perspective_transform(thresh)

    if left_line.do_slide_window() or right_line.do_slide_window():
        left_fit, right_fit = find_lane_lines_sliding_window(warped, plot)
    else:
        left_fit, right_fit = find_lane_lines_using_previous_fit(warped, left_line.current_fit, right_line.current_fit, plot)

    valid_lane = check_lane(left_fit, right_fit)
    left_line.add_fit(left_fit, valid_lane)
    right_line.add_fit(right_fit, valid_lane)

    # Draw the best fit
    if to_draw_img != None:
        result = draw_lane(img, warped, left_line.best_fit, right_line.best_fit)
    else:
        result = draw_lane(img, warped, left_line.best_fit, right_line.best_fit)
    draw_radius_of_curvature(result,
            (left_line.radius_of_curvature+right_line.radius_of_curvature)/2)
    draw_vehicle_position(result, left_line.best_fit, right_line.best_fit)

    if dump:
        filename = "./video_frames_images/" + str(left_line.frame_counter) + ".jpg"
        mpimg.imsave(filename, img)
        filename = "./video_frames_processed_images/" + str(left_line.frame_counter) + ".jpg"
        mpimg.imsave(filename, result)

    if plot == True:
        plt.imshow(result)
        plt.show(block=True)	

    return result

def check_width(left_fit, right_fit):
    # Check lane width
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    diff = right_fitx - left_fitx
    min_width = np.min(diff)
    max_width = np.max(diff)
    if min_width < 550 or max_width > 850: # 550 <= valid lane width <= 850
        return False

    return True

def check_lane(left_fit, right_fit):
    if left_line.best_fit == None or right_line.best_fit == None:
        return True # Take whatever we have right now

    if not check_width(left_fit, right_fit):
        return False

    if not left_line.check_line(left_fit):
        return False

    if not right_line.check_line(right_fit):
        return False
    
    return True

def process_video(in_name, out_name, plot = False, dump = False):
    from moviepy.editor import VideoFileClip
    output1 = './project_video_processed.mp4'
    clip1 = VideoFileClip("./project_video.mp4")
    processed_clip1 = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    processed_clip1.write_videofile(output1, audio=False)

processor = ImageProcessor()
left_line = Line()
right_line = Line()

#MODE = 2 # 1 - Images, 2 - Videos

#processor = ImageProcessor()

#if MODE == 1:
    #run_all_test_images('./test_images', plot=True, debug=False)
    ##run_all_test_images('./my_test_images/other_types_test_images/', plot=True, debug=True)
    ##run_all_test_images('./my_test_images', plot=True, debug=True)
#elif MODE == 2:
    #left_line = Line()
    #right_line = Line()
    #process_video('./project_video.mp4', 'project_video_processed.mp4')

