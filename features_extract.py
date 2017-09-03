import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog
import matplotlib.pyplot as plt

# Helper function for displaying two images in parallel to compare.
def plotTwo(image, title, saveName=''):
    fig, axis = plt.subplots(1, 2, figsize=(12, 10))
    fig.tight_layout()
    for i in range(2):
        if len(image[i].shape) == 2:
            axis[i].imshow(image[i], cmap='gray')
        else:
            axis[i].imshow(image[i])
        axis[i].set_title(title[i])
    if saveName != '':
        plt.savefig(saveName, bbox_inches='tight')
    plt.show(block=True)

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32), plot = False, name = ''):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 

    if plot:
        plotTwo((img, cv2.resize(img, size)), (name, 'bin_spatial'), name+'-bin_spatial')

    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256), plot = False, name = ''):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    if plot:
        # Generating bin centers
        bin_edges = channel1_hist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.bar(bin_centers, channel1_hist[0])
        plt.xlim(0, 256)
        plt.title('Channel1 Histogram')
        plt.subplot(132)
        plt.bar(bin_centers, channel2_hist[0])
        plt.xlim(0, 256)
        plt.title('Channel2 Histogram')
        plt.subplot(133)
        plt.bar(bin_centers, channel3_hist[0])
        plt.xlim(0, 256)
        plt.title('Channel3 Histogram')
        fig.tight_layout()
        plt.savefig(name+'-color_hist', bbox_inches='tight')
        plt.show(block=True)

    # Return the feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Apply color convertion based on given color space.
def convert_color(image, cspace, from_cspace, plot = False, name = ''):
    if cspace != 'RGB':
        if cspace == 'HSV':
            if from_cspace == 'from_BGR':
                cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            else:
                cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            if from_cspace == 'from_BGR':
                cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            else:
                cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            if from_cspace == 'from_BGR':
                cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            else:
                cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            if from_cspace == 'from_BGR':
                cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            else:
                cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            if from_cspace == 'from_BGR':
                cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            else:
                cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        if plot:
            plotTwo((image, cvt_image), ('original', 'convert_color'), name+'-convert_color')
        return cvt_image
    else:
        return image


# Define a function to extract features from a list of images
# Have this function call bin_spatial(), color_hist(), and get_hog_features()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0,256),
		     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img_name in imgs:
        # Read in each one by one
        image = cv2.imread(img_name)
        # apply color conversion
        feature_image = convert_color(image, cspace, 'from_BGR');

	# Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(np.concatenate([spatial_features, hist_features, hog_features]))
    # Return list of feature vectors
    return features
