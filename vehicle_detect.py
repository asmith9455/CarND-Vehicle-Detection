import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib.image as mpimg

import glob

from vehicle_features import *

from heat_map import *

import pickle

# specify parameters that configure the behaviour of the code below, including:
# - mode (train/video/photo mode)
# - initialize classifier object that is filled out differently depending on the combination of mode flags. For example, if training mode and video mode are active, I assume that the user wishes to use the newly generated classifier, rather than read on from disk.
# - the X_scaler object is initialized here for the same reason as clf (the previous bullet point), but represents an object that can normalize the input features
# - the names of the classifier to read from disk, if train_mode is false and one of photo or video_mode is true.
# - the name of the test video to read (also used to generate the output file name)

train_mode = False
video_mode = True
# photo_mode = False

clf = None
X_scaler = None

model_name = "model_v0.6"
video_name = "project_video"


# below is a specification of the sliding windows used for detection

# height (rows), width (cols), 
# scan_height (rows), scan_width (cols), 
# first_row, last_row, 
# first_col, last_col
# keep_prob
windows = \
[
	(128, 128, 64, 64, 0.65, 0.9, 0.5, 1.0, 1.0),
	(64, 64, 32, 32, 0.55, 0.7, 0.5, 1.0, 1.0),
	(96, 96, 48, 48, 0.55, 0.7, 0.5, 1.0, 1.0),

	(128, 128, 64, 64, 0.65, 0.9, 0.5, 1.0, 1.0),
	(128, 128, 48, 48, 0.7, 1.0, 0.5, 1.0, 1.0),
	(64, 64, 32, 32, 0.55, 0.7, 0.5, 1.0, 0.5),
	
	(96, 96, 48, 48, 0.55, 0.7, 0.5, 1.0, 1.0),

	(96, 96, 36, 36, 0.54, 0.7, 0.5, 1.0, 1.0),

	(96, 96, 24, 24, 0.55, 0.7, 0.5, 1.0, 1.0)
]

if train_mode:
	# if in training mode, read data set file names and make data within these  files available to the training/testing functinos

	vehicle_files = glob.glob('../data/P5/vehicles/vehicles/KITTI_extracted/*.png', recursive=True)
	vehicle_time_series_files = glob.glob('../data/P5/vehicles/vehicles/GTI_*/*.png', recursive=True)
	
	nonvehicle_files = glob.glob('../data/P5/non-vehicles/non-vehicles/Extras/*.png', recursive=True)
	nonvehicle_time_series_files = glob.glob('../data/P5/non-vehicles/non-vehicles/GTI/*.png', recursive=True)

	vehicle_files = vehicle_files + vehicle_time_series_files
	nonvehicle_files = nonvehicle_files + nonvehicle_time_series_files

	print('number of vehicle files is ', len(vehicle_files))
	print('number of non-vehicle files is ', len(nonvehicle_files))

	print("loading vehicle features from disk...")

	features_vehicles = extract_features(vehicle_files, cspace="HLS", hog_channel="ALL")

	print("finished loading vehicle features from disk")

	print("loading non-vehicle features from disk...")

	features_nonvehicles = extract_features(nonvehicle_files, cspace="HLS", hog_channel="ALL")

	print("finished loading non-vehicle features from disk")

	X = np.vstack((features_vehicles, features_nonvehicles)).astype(np.float64)

	# generate labels for the whole data set
	#1 means vehicle, 0 means "not a vehicle"
	y = np.hstack((np.ones(len(features_vehicles)), np.zeros(len(features_nonvehicles))))

	rand_state = np.random.randint(0, 100)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand_state)

	print("finished splitting into training and testing sets")

	# initialize and fit the StandardScaler for the training data
	X_scaler = StandardScaler().fit(X_train)

	print("performing the normalization (based on training data) on both the training and test data")

	# apply the StandardScaler to both the training and test data.
	X_train = X_scaler.transform(X_train)
	X_test = X_scaler.transform(X_test)

	# print("finding optimal decision tree...")

	# clf_tree = tree.DecisionTreeClassifier()

	# clf_tree.fit(X_train, y_train)

	# print("found optimal decision tree")

	print("finding optimal support vector machine...")

	# I experiemented with a set of parameters that are automatically 'grid searched' by GridSearchCV, but decided against using this concept in the end.
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

	# svr = svm.LinearSVC()
	# svr = 
	# I chose a linear kernel and C values of 1e-3
	clf = svm.SVC(kernel='linear', C=1e-3, verbose=2) #GridSearchCV(svr, parameters)

	clf.fit(X_train, y_train)
	
	print("found optimal support vector machine")

	# now that the classifier has been fit to the training data, evaluate its accuracy on both the training and test data

	print("predicting on test and train set...")


	y_pred_test = clf.predict(X_test)
	y_pred_train = clf.predict(X_train)

	# y_pred_test_tree = clf_tree.predict(X_test)
	# y_pred_train_tree = clf_tree.predict(X_train)

	print("finished predicting on test set")

	acc = \
	{
		'train_svm: ': accuracy_score(y_train, y_pred_train),
		'test_svm: ': accuracy_score(y_test, y_pred_test),
		
		# 'train_tree: ': accuracy_score(y_train, y_pred_train_tree),
		# 'test_tree: ': accuracy_score(y_test, y_pred_test_tree)
	}

	print("accuracies:\n", acc)
	
	# write the newly fit model and StandardScaler to disk for later reuse without re-fitting

	with open(model_name, 'wb') as output:
		pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
	
	with open(model_name + "_normalizer", 'wb') as output:
		pickle.dump(X_scaler, output, pickle.HIGHEST_PROTOCOL)


if video_mode:
	# in this mode, process as video with the name specified above
	# if we didn't train the classifier above, read the clf and StandardScaler from disk.

	from moviepy.editor import VideoFileClip

	if not(train_mode):
		with open(model_name, 'rb') as ip_file:
			clf = pickle.load(ip_file)
		with open(model_name + "_normalizer", 'rb') as ip_file:
			X_scaler = pickle.load(ip_file)

	hmap = HeatMap((720, 1280, 3))

	def video_image(img):

		# detect objects in the image using all sliding windows.

		bboxes, allboxes = detect_objects(img, clf, X_scaler, windows)

		# bboxes2, allboxes2 = detect_objects_multi_scale(img, clf_tree, X_scaler, min_size=(100,100), max_size=(201,201), step_size=(100,100))

		# form the image with the bounding boxes drawn on it 
		draw_img = draw_bboxes(img, bboxes, allboxes)
		
		# add the detections to the heatmap
		hmap.add_boxes(bboxes)

		# final detections heatmap threshold (experimentally determined)
		hmap_thresh = 650

		# generate the binary map of detections
		bin_map = np.zeros(hmap.shape, dtype=np.uint8)
		bin_map[hmap.map[:,:,0] > hmap_thresh] = np.array([255,0,0])

		# cv2.imshow("heatmap", hmap.map)

		# cv2.waitKey(100)

		# draw_img = draw_bboxes(img, bboxes2, allboxes2)

		# overlay the binary map of detected objects onto the original image 
		draw_img = cv2.addWeighted(draw_img, 1.0, bin_map, 0.5, 0.0)

		# find the boxes that represent vehicle detections from the binary map  
		bin_img_for_contours = bin_map[:,:,0]

		mod_img, contours, hierarchy = cv2.findContours(bin_img_for_contours.copy(), 1, 2)

		draw_img_2 = img.copy()

		# require an area of at least 600 px^2 for the detected box

		area_thresh = 600  #area theshold in pixels squared - to reduce false positives

		# draw the final detections on the output video
		for c in contours:
			x, y, w, h = cv2.boundingRect(c)

			if (w * h < area_thresh):
				continue

			cv2.rectangle(draw_img,(x,y),(x+w,y+h),(0,0,255),2)
			cv2.rectangle(draw_img_2,(x,y),(x+w,y+h),(0,0,255),2)

		# put the image with vehicle detections on top, the sliding windows and pre-heatmap detections on the bottom left, and the heatmap on the bottom right, then return this image

		draw_img_resize = cv2.resize(draw_img, (0,0), fx=0.5, fy=0.5)

		hmap_resize = cv2.resize((hmap.map * 255 / np.max(hmap.map)).astype(np.uint8), (0,0), fx=0.5, fy=0.5)

		bottom = np.hstack((draw_img_resize, hmap_resize))

		return np.vstack((draw_img_2, bottom))

	test_clip = VideoFileClip(video_name + ".mp4")
	output_vid = test_clip.fl_image(video_image)
	output_vid.write_videofile(video_name + "_output.mp4")

# if photo_mode:

# 	files = glob.glob('test_images/*.jpg')

# 	if not(train_mode):
# 		with open(model_name, 'rb') as ip_file:
# 			clf = pickle.load(ip_file)
# 		with open(model_name + "_normalizer", 'rb') as ip_file:
# 			X_scaler = pickle.load(ip_file)

# 	for file in files:
# 		img = mpimg.imread(file)

# 		bboxes, allboxes = detect_objects(img, clf, X_scaler, windows)
# 		draw_img = draw_bboxes(img, bboxes, allboxes)

# 		cv2.imshow("detections", cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))

# 		cv2.waitKey(0)

	