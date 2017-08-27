import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from features_extract import convert_color, get_hog_features, bin_spatial, color_hist
import glob

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, colorspace, orient,
              pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, cspace=colorspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    if hog_channel == 'ALL':
       channel = [ctrans_tosearch[:,:,0], ctrans_tosearch[:,:,1], ctrans_tosearch[:,:,2]]
    else:
       channel = [ctrans_tosearch[:,:,hog_channel]]

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
            #hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            #hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            #hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            #hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img

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
print('Using:', 'spatial binning of', spatial, histbin,'histogram bins', orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')


testFiles = glob.glob('test_images/*.jpg')
example = testFiles[0]
image = mpimg.imread(example)
ystart = 400
ystop = 656
scale = 1
draw_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial, histbin)
plt.imshow(draw_img)
plt.show(block=True)
