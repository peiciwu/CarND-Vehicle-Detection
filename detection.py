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
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog = []
    for ch in channel:
        hog.append(get_hog_features(ch, orient, pix_per_cell, cell_per_block, feature_vec=False))
    
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
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
            test_prediction = svc.predict(test_features)
            #test_prediction = int(svc.decision_function(test_features) > 0.8)
            
            if test_prediction == 1:
                #print('decision_function:', svc.decision_function(test_features))
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
        print("add_heat, box:", box)

    print("after add, min:", np.min(heatmap), "max:", np.max(heatmap))

    # Return updated heatmap
    return heatmap

def remove_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Decrease -= 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] -= 1
        print("remove_heat, box:", box)
        if np.min(heatmap) < 0:
            print("found negative, argmin0:", np.argmin(heatmap, 0), "argmin1:",
                    np.argmin(heatmap,1))

    print("after remove, min:", np.min(heatmap), "max:", np.max(heatmap))

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

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
        filter_img = draw_labeled_bboxes(np.copy(image), labels)

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
    global heatmap, num_frames, recent_heats
    bbox_list = []
    for scale in [1, 1.25, 1.75]:
        bbox_list += find_cars(img, ystart, ystop, scale, svc, X_scaler,
                colorspace, orient, pix_per_cell, cell_per_block, hog_channel,
                spatial_size=(spatial, spatial), hist_bins=histbin)
    #for ystart, ystop, scale in [(y1, y2, 1), (y1, y3, 1.25), (y2, y3, 1.75)]:
        #bbox_list += find_cars(img, ystart, ystop, scale, svc, X_scaler,
                #colorspace, orient, pix_per_cell, cell_per_block, hog_channel,
                #spatial_size=(spatial, spatial), hist_bins=histbin)

    if len(recent_heats) == num_frames:
        heatmap = remove_heat(heatmap, recent_heats.popleft())

    #correct_heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    #if np.allclose(correct_heatmap, heatmap) == False:
        #print('heatamp after remove is not all zeros')
    #correct_heatmap = add_heat(correct_heatmap, bbox_list)

    # Add heat to each box in bounding-box list
    heatmap = add_heat(heatmap, bbox_list)
    recent_heats.append(bbox_list)

    #if np.array_equal(correct_heatmap, heatmap) == False:
        #print('heatamp after add is not the same')

    # Apply threshold to help remove false positives
    thresh_map = apply_threshold(heatmap, 2)
    # Find final boxes from heatmap using label function
    labels = label(thresh_map)

    #lane_img = process_image(img)
    #vehicle_img = draw_labeled_bboxes(lane_img, labels)
    vehicle_img = draw_labeled_bboxes(img, labels)
    return vehicle_img

def process_video(in_name, out_name):
    from moviepy.editor import VideoFileClip
    output1 = out_name
    clip1 = VideoFileClip(in_name)
    processed_clip1 = clip1.fl_image(process_image_with_vehicle_detection) #NOTE: this function expects color images!!
    processed_clip1.write_videofile(output1, audio=False)

# Load the saved model and conifg.
saved = joblib.load('./model5.sav')
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

#run_images()
y1 = 400
y2 = 450
y3 = 650

# For processing video
#heatmap = np.zeros((720, 1280), dtype=np.float) # FIXME. debug
heatmap = np.zeros_like(mpimg.imread('test_images/test1.jpg')[:,:,0]).astype(np.float)
num_frames = 1
recent_heats = collections.deque(maxlen=num_frames)
#process_video('./test_videos/1.mp4', './test_videos/1_processed_5.mp4')
#process_video('./test_videos/2.mp4', './test_videos/2_processed_1.mp4')
#process_video('./test_videos/test_video_1.mp4', './test_videos/test_video_1_processed_4.mp4')
process_video('./test_video.mp4', './test_video_processed_6.mp4')
