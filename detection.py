import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from features_extract import convert_color, get_hog_features, bin_spatial, color_hist
import glob
from process_image import process_image
import collections

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, colorspace, orient,
              pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins):
    bbox_list = [] # Return a list of detected bounding boxes
    img = img.astype(np.float32)/255
    
    #FIXME: pw debug
    #img_tosearch = img[ystart:ystop,:,:]
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, cspace=colorspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    if hog_channel == 'ALL':
       channel = [ctrans_tosearch[:,:,0], ctrans_tosearch[:,:,1], ctrans_tosearch[:,:,2]]
    else:
       channel = [ctrans_tosearch[:,:,int(hog_channel)]]

    # Define blocks and steps as above
    nxblocks = (channel[0].shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (channel[0].shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    #cells_per_step = 2  # Instead of overlap, define how many cells to step
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog = []
    for ch in channel:
        hog.append(get_hog_features(ch, orient, pix_per_cell, cell_per_block, feature_vec=False))
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat = []
            for h in hog:
                hog_feat.append(h[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
            if hog_channel == 'ALL':
                hog_features = np.hstack((hog_feat[0], hog_feat[1], hog_feat[2]))
            else:
                hog_features = hog_feat[0]

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_prediction = svc.predict(test_features)
            #test_prediction = int(svc.decision_function(test_features) > 0.05)
            #test_prediction = int(svc.decision_function(test_features) > 0.2)
            #test_prediction = int(svc.decision_function(test_features) > 0.3)
            #test_prediction = int(svc.decision_function(test_features) > 0.5)
            #test_prediction = int(svc.decision_function(test_features) > 0.6)
            test_prediction = int(svc.decision_function(test_features) > 0.7)
            
            if test_prediction == 1:
                #print('decision_function:', np.min(svc.decision_function(test_features)))
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox_list.append(((int(xbox_left), int(ytop_draw+ystart)), (int(xbox_left+win_draw), int(ytop_draw+win_draw+ystart))))
                
    return bbox_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def get_labeled_bboxes(labels):
    bbox_list = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
    # Return the image
    return bbox_list

def run_images():
    testFiles = glob.glob('test_images/*.jpg')
    for img_file in testFiles:
        image = mpimg.imread(img_file)
        bbox_list = []
        for scale in [1, 1.5, 1.75]:
            bbox_list += find_cars(image, ystart, ystop, scale, svc, X_scaler,
                    colorspace, orient, pix_per_cell, cell_per_block, hog_channel,
                    spatial_size=(spatial, spatial), hist_bins=histbin)
        draw_img = np.copy(image)
        for bbox in bbox_list:
            cv2.rectangle(draw_img, bbox[0], bbox[1],(0,0,255),6) 
        plt.title('Detected image')
        plt.imshow(draw_img)
        plt.show(block=True)

        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        # Add heat to each box in bounding-box list
        heatmap = add_heat(heatmap, bbox_list)
        # Apply threshold to help remove false positives
        heatmap = apply_threshold(heatmap, 2)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        filter_img = np.copy(image)
        labeled_bbox_list = get_labled_bboxes(labels)
        for bbox in labeled_bbox_list:
            cv2.rectangle(filter_img, bbox[0], bbox[1],(0,0,255),6) 

        print(labels[1], 'cars found')
        plt.title('label image')
        plt.imshow(labels[0], cmap='gray')
        plt.show(block=True)

        plt.title('Filtered Detected image')
        plt.imshow(filter_img)
        plt.show(block=True)

        # FIXME: for drawing lane lines
        #lane_img = process_image(image, to_draw_img = filter_img)
        #plt.title('Filtered Detected image with lane lines')
        #plt.imshow(lane_img)
        #plt.show(block=True)

def process_image_with_vehicle_detection(img):
    global recent_heats, frame_counter, labeled_bbox_list

    if frame_counter%2 == 0:
        bbox_list = []
        #for scale in [1.25, 1.75]:
            #bbox_list += find_cars(img, ystart, ystop, scale, svc, X_scaler,
                    #colorspace, orient, pix_per_cell, cell_per_block, hog_channel,
                    #spatial_size=(spatial, spatial), hist_bins=histbin)
        #for ystart, ystop, scale in [(y1, y3, 1.75), (y1, y4, 1.25), (y1, y3, 0.75), (y1, y3, 2.5)]:
        #for ystart, ystop, scale in [(y1, y3, 1.75), (y1, y4, 1.25), (y1, y3,0.75), (y1, y3, 0.5)]:
        #for ystart, ystop, scale in [(y1, y4, 1.25), (y1, y4, 1), (y1, y3, 1.75)]:
        #for ystart, ystop, scale in [(y1, y4, 1.25), (425, 525, 1.5)]:
        for ystart, ystop, scale in [(y1, y4, 1.25), (425, 525, 1.5), (425, 525, 1.75)]:
            bbox_list += find_cars(img, ystart, ystop, scale, svc, X_scaler,
                    colorspace, orient, pix_per_cell, cell_per_block, hog_channel,
                    spatial_size=(spatial, spatial), hist_bins=histbin)

        # Add heat to each box in bounding-box list
        cur_heat = np.zeros_like(img[:,:,0]).astype(np.float)
        cur_heat = add_heat(cur_heat, bbox_list)
        recent_heats.append(cur_heat)

        heatmap = sum(recent_heats)

        # Apply threshold to help remove false positives
        #thresh_map = apply_threshold(heatmap, 5)
        thresh_map = apply_threshold(heatmap, 3)
        # Find final boxes from heatmap using label function
        labels = label(thresh_map)
        labeled_bbox_list = get_labeled_bboxes(labels)

    img = process_image(img)
    for bbox in labeled_bbox_list:
        cv2.rectangle(img, bbox[0], bbox[1],(0,0,255),6) 
    frame_counter+=1
    return img

def process_video(in_name, out_name):
    from moviepy.editor import VideoFileClip
    output1 = out_name
    clip1 = VideoFileClip(in_name)
    processed_clip1 = clip1.fl_image(process_image_with_vehicle_detection) #NOTE: this function expects color images!!
    processed_clip1.write_videofile(output1, audio=False)

# FIXME: pw debug
#image = mpimg.imread('./test_videos/4.jpg')
#cv2.line(image, (0, 400), (1280, 400), (0, 0, 255), 3)
#cv2.line(image, (0, 450), (1280, 450), (0, 0, 255), 3)
#cv2.line(image, (0, 500), (1280, 500), (0, 0, 255), 3)
#cv2.line(image, (0, 550), (1280, 550), (0, 0, 255), 3)
#cv2.line(image, (0, 600), (1280, 600), (0, 0, 255), 3)
#plt.imshow(image)
#plt.show(block=True)
#quit()

# Load the saved model and conifg.
saved = joblib.load('./model4.sav')
svc = saved['model']
X_scaler = saved['X_scaler']
config = saved['config']
# Set parameters based on config
colorspace = config['colorspace']
spatial = config['spatial']
histbin = config['histbin']
orient = config['orient']
pix_per_cell = config['pix_per_cell']
cell_per_block = config['cell_per_block']
hog_channel = config['hog_channel']
print('Using colorspace:', colorspace, 'spatial binning of', spatial, histbin,'histogram bins', orient,'orientations',pix_per_cell,
    'pixels per cell', cell_per_block,'cells per block', 'and hog channel', hog_channel)

ystart = 400
ystop = 650

y1 = 400
y2 = 500
y3 = 550
y4 = 650

#run_images()

# For processing video
num_frames = 3
recent_heats = collections.deque(maxlen=num_frames)
frame_counter = 0
labeled_bbox_list = None

# For lane lines
processor = ImageProcessor()
left_line = Line()
right_line = Line()

#process_video('./test_videos/1.mp4', './test_videos/1_processed_21_175.mp4')
#process_video('./test_videos/2.mp4', './test_videos/2_processed_4.mp4')
#process_video('./test_videos/3.mp4', './test_videos/3_processed_4.mp4')
#process_video('./test_videos/4.mp4','./test_videos/4_processed_22_125_150_3_2_06.mp4')
#process_video('./test_videos/5.mp4', './test_videos/5_processed_2.mp4')
process_video('./test_videos/test_video_1.mp4', './test_videos/test_video_1_processed_17.mp4')
#process_video('./test_video.mp4', './test_video_processed_7.mp4')
#process_video('./project_video.mp4', './project_video_processed_5.mp4')
