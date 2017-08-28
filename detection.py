import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from features_extract import convert_color, get_hog_features, bin_spatial, color_hist
import glob
from process_image import process_image

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
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            #test_prediction = svc.predict(test_features)
            test_prediction = int(svc.decision_function(test_features) > 0.6)
            
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

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in bounding-box list
    heat = add_heat(heat, bbox_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
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
