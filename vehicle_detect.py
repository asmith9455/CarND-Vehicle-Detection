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

train_mode = False
video_mode = True
photo_mode = False

clf = None
X_scaler = None

model_name = "model_v0.6"
video_name = "project_video"

# height (rows), width (cols), 
# scan_height (rows), scan_width (cols), 
# first_row, last_row, 
# first_col, last_col
# keep_prob
windows = \
[
	# (	256, 256, 
	# 	16, 16, 
	# 	0.5, 0.75, 
	# 	0.0, 1.0),

	(128, 128, 64, 64, 0.5, 0.7, 0.5, 1.0, 1.0), #VERY GOOD ARRANGEMENT
	(64, 64, 32, 32, 0.5, 0.7, 0.5, 1.0, 1.0),
	(96, 96, 48, 48, 0.5, 0.7, 0.5, 1.0, 1.0),

	(128, 128, 64, 64, 0.5, 0.7, 0.5, 1.0, 1.0), #VERY GOOD ARRANGEMENT
	(64, 64, 32, 32, 0.5, 0.7, 0.5, 1.0, 1.0),
	(96, 96, 48, 48, 0.5, 0.7, 0.5, 1.0, 1.0)
	# (128, 128, 32, 32, 0.5, 0.75, 0.0, 1.0),
	# (32, 32, 16, 16, 0.52, 0.65, 0.7, 1.0, 1.0),
]

if train_mode:
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

	#1 means vehicle, 0 means "not a vehicle"
	y = np.hstack((np.ones(len(features_vehicles)), np.zeros(len(features_nonvehicles))))

	rand_state = np.random.randint(0, 100)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand_state)

	print("finished splitting into training and testing sets")

	X_scaler = StandardScaler().fit(X_train)

	print("performing the normalization (based on training data) on both the training and test data")

	X_train = X_scaler.transform(X_train)
	X_test = X_scaler.transform(X_test)

	# print("finding optimal decision tree...")

	# clf_tree = tree.DecisionTreeClassifier()

	# clf_tree.fit(X_train, y_train)

	# print("found optimal decision tree")

	print("finding optimal support vector machine...")

	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

	# svr = svm.LinearSVC()
	# svr = 
	clf = svm.SVC(kernel='linear', C=1e-3, verbose=2) #GridSearchCV(svr, parameters)

	clf.fit(X_train, y_train)
	
	print("found optimal support vector machine")



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

	hmap = HeatMap((720, 1280, 3))

	def video_image(img):
		bboxes, allboxes = detect_objects(img, clf, X_scaler, windows)

		# bboxes2, allboxes2 = detect_objects_multi_scale(img, clf_tree, X_scaler, min_size=(100,100), max_size=(201,201), step_size=(100,100))
		draw_img = draw_bboxes(img, bboxes, allboxes)
		

		hmap.add_boxes(bboxes)

		hmap_thresh = 600

		bin_map = np.zeros(hmap.shape, dtype=np.uint8)
		bin_map[hmap.map[:,:,0] > hmap_thresh] = np.array([255,0,0])

		# cv2.imshow("heatmap", hmap.map)

		# cv2.waitKey(100)

		# draw_img = draw_bboxes(img, bboxes2, allboxes2)

		draw_img = cv2.addWeighted(draw_img, 1.0, bin_map, 0.5, 0.0)

		# find contours

		bin_img_for_contours = bin_map[:,:,0]

		mod_img, contours, hierarchy = cv2.findContours(bin_img_for_contours.copy(), 1, 2)

		draw_img_2 = img.copy()

		area_thresh = 600  #area theshold in pixels squared - to reduce false positives

		for c in contours:
			x, y, w, h = cv2.boundingRect(c)

			if (w * h < area_thresh):
				continue

			cv2.rectangle(draw_img,(x,y),(x+w,y+h),(0,0,255),2)
			cv2.rectangle(draw_img_2,(x,y),(x+w,y+h),(0,0,255),2)

		draw_img_resize = cv2.resize(draw_img, (0,0), fx=0.5, fy=0.5)

		hmap_resize = cv2.resize((hmap.map * 255 / np.max(hmap.map)).astype(np.uint8), (0,0), fx=0.5, fy=0.5)

		bottom = np.hstack((draw_img_resize, hmap_resize))

		return np.vstack((draw_img_2, bottom))

	test_clip = VideoFileClip(video_name + ".mp4").subclip(5,40)
	output_vid = test_clip.fl_image(video_image)
	output_vid.write_videofile(video_name + "_output.mp4")

if photo_mode:

	files = glob.glob('test_images/*.jpg')

	if not(train_mode):
		with open(model_name, 'rb') as ip_file:
			clf = pickle.load(ip_file)
		with open(model_name + "_normalizer", 'rb') as ip_file:
			X_scaler = pickle.load(ip_file)

	for file in files:
		img = mpimg.imread(file)

		bboxes, allboxes = detect_objects(img, clf, X_scaler, windows)
		draw_img = draw_bboxes(img, bboxes, allboxes)

		cv2.imshow("detections", cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))

		cv2.waitKey(0)

	