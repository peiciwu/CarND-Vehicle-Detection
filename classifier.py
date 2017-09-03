import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import time

from features_extract import convert_color, bin_spatial, color_hist, get_hog_features, extract_features, plotTwo

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype

    return data_dict

# Read in cars and non-car images
cars = glob.glob('classifier_inputs/vehicles/*/*/*.png')
notcars = glob.glob('classifier_inputs/non-vehicles/*/*/*.png')

data_look(cars, notcars)

# Parameters for features extraction
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial = 16 # number of spatial bins
histbin = 8 # number of histogram bins
orient = 8 # numbre of HOG orientations bins (typically between 6 and 12)
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or 'ALL'

# Test
#for inName, outName in [(cars[100], 'car'), (notcars[100], 'notcar')]:
#    img = cv2.imread(inName)
#    img = convert_color(img, colorspace, 'from_BGR', plot = True, name = outName)
#    bin_spatial(img, (spatial, spatial), plot = True, name = outName)
#    color_hist(img, histbin, (0, 256), plot = True, name = outName)
#    hog_features, hog_img = get_hog_features(img[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
#    plotTwo((img[:,:,0], hog_img), ('original', 'hog_image'), outName+'-hog_image')

t=time.time()
car_features = extract_features(cars, cspace=colorspace, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0,256),
			        orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0,256),
			           orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using colorspace:', colorspace, 'spatial binning of', spatial, histbin,'histogram bins', orient,'orientations',pix_per_cell,
    'pixels per cell', cell_per_block,'cells per block', 'and hog channel', hog_channel)
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# Save model and config
config = dict(colorspace=colorspace, spatial=spatial, histbin=histbin, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
joblib.dump({'model':svc, 'X_scaler':X_scaler, 'config':config}, 'model8.sav')
