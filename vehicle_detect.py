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

import pickle

train_mode = False
video_mode = True
photo_mode = False

clf = None
X_scaler = None

model_name = "model_v0.4"
video_name = "project_video"

if train_mode:
	vehicle_files = glob.glob('../data/P5/vehicles/vehicles/KITTI_extracted/*.png', recursive=True)
	nonvehicle_files = glob.glob('../data/P5/non-vehicles/non-vehicles/Extras/*.png', recursive=True)

	vehicle_files = vehicle_files[::1]
	nonvehicle_files = nonvehicle_files[::1]

	print('number of vehicle files is ', len(vehicle_files))
	print('number of non-vehicle files is ', len(nonvehicle_files))

	print("loading features from disk...")

	features_vehicles = extract_features(vehicle_files, cspace="HLS", hog_channel=2)
	features_nonvehicles = extract_features(nonvehicle_files, cspace="HLS", hog_channel=2)

	print("finished loading features from disk")

	X = np.vstack((features_vehicles, features_nonvehicles)).astype(np.float64)

	#1 means vehicle, 0 means "not a vehicle"
	y = np.hstack((np.ones(len(features_vehicles)), np.zeros(len(features_nonvehicles))))

	rand_state = np.random.randint(0, 100)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand_state)

	print("finished splitting into training and testing sets")

	X_scaler = StandardScaler().fit(X_train)

	print("performing the normalization (based on training data) on both the training and test data")

	X_train = X_scaler.transform(X_train)
	X_test = X_scaler.transform(X_test)

	print("finding optimal support vector machine...")

	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

	# svr = svm.LinearSVC()
	# svr = 
	clf = svm.SVC(kernel='linear', C=1e-3) #GridSearchCV(svr, parameters)

	clf.fit(X_train, y_train)

	print("found optimal support vector machine")

	print("predicting on test and train set...")

	y_pred_test = clf.predict(X_test)
	y_pred_train = clf.predict(X_train)

	print("finished predicting on test set")

	acc_test = accuracy_score(y_test, y_pred_test)
	acc_train = accuracy_score(y_train, y_pred_train)

	print("accuracy (test): ", acc_test)
	print("accuracy (train): ", acc_train)

	print("first 10 predictions")

	first_pred = clf.predict(X_test[-9:])

	print("labels (pred):\n", first_pred)

	print("labels (test):\n", y_test[-9:])

	s = pickle.dumps(clf)

	

	with open(model_name, 'wb') as output:
		pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
	
	with open(model_name + "_normalizer", 'wb') as output:
		pickle.dump(X_scaler, output, pickle.HIGHEST_PROTOCOL)


if video_mode:

	from moviepy.editor import VideoFileClip

	if not(train_mode):
		with open(model_name, 'rb') as ip_file:
			clf = pickle.load(ip_file)
		with open(model_name + "_normalizer", 'rb') as ip_file:
			X_scaler = pickle.load(ip_file)
		
	def video_image(img):
		bboxes, allboxes = detect_objects_multi_scale(img, clf, X_scaler, min_size=(100,100), max_size=(201,201), step_size=(100,100))

		draw_img = draw_bboxes(img, bboxes, allboxes)

		return draw_img

	test_clip = VideoFileClip(video_name + ".mp4")
	output_vid = test_clip.fl_image(video_image)
	output_vid.write_videofile(video_name + "_output.mp4")

if photo_mode:

	photo = mpimg.imread("../data/P5//")

	if not(train_mode):
		with open(model_name, 'rb') as ip_file:
			clf = pickle.load(ip_file)
		with open(model_name + "_normalizer", 'rb') as ip_file:
			X_scaler = pickle.load(ip_file)
	
	bboxes, allboxes = detect_objects_multi_scale(img, clf, X_scaler, min_size=(100,100), max_size=(201,201), step_size=(100,100))

	out_img = draw_boxes(img, bboxes, allboxes)


	