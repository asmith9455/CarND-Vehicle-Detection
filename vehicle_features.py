import matplotlib.image as mpimg
from skimage.feature import hog

import numpy as np

import cv2



def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)

        return features

def extract_features_img(image, cspace='RGB', orient=9, 
                        pix_per_cell=16, cell_per_block=2, hog_channel=0):
    # apply color conversion if other than 'RGB'
    if cspace == 'HSV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)      

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
    
    # extract colour information

    histo0, bin_ranges0 = np.histogram(feature_image[:,:,0], bins=8, density=True)
    histo1, bin_ranges1 = np.histogram(feature_image[:,:,1], bins=8, density=True)
    histo2, bin_ranges2 = np.histogram(feature_image[:,:,2], bins=8, density=True)
    


    # Append the new feature vector to the features list
    return np.concatenate((histo0, histo1, histo2, hog_features))

def extract_features_imgs(images, cspace='RGB', orient=9, 
                        pix_per_cell=16, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in images:
        # Read in each one by one
        hog_features = extract_features_img(img, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        features.append(hog_features)
    # Return list of feature vectors
    features_np =  np.array(features)

    print("features shape: ", features_np.shape)

    return features_np

#use the extract features function developed in the course to pull out the colour and HOG information from the images
def extract_features(img_filepaths, cspace='RGB', orient=9, pix_per_cell=16, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in img_filepaths:
        # Read in each one by one
        image = mpimg.imread(file)
        img_features = extract_features_img(image, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

        #perform horizontal flip of image - should have the same class
        #img_features2 = extract_features_img(image[:,::-1,:], cspace=cspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
        
        features.append(img_features)
        #features.append(img_features2)
    # Return list of feature vectors
    return np.array(features)


def detect_objects(image, clf, feature_scaler, windows):

    #args *_shape is (rows, columns) aka (height, width)

    imgs = []
    bboxes = []

    for window in windows:

        first_row = int(float(image.shape[0]) * window[4])
        first_col = int(float(image.shape[1]) * window[6])

        last_row = int(float(image.shape[0]) * window[5])
        last_col = int(float(image.shape[1]) * window[7])

        window_width = window[1]     #columns
        window_height = window[0]    #rows

        scan_width = window[3]
        scan_height = window[2]

        keep_prob = window[8]

        #todo: enable denser sampling around a list of input locations (tracked objects)

        for start_row in range(first_row, last_row, scan_height):
            
            # enable random fluxuations in the position of the scan boxes 
            
            start_row = int(start_row + (scan_height / 2.0) * (np.random.uniform() - 0.5))

            end_row = start_row + window_height

            if (end_row > image.shape[0] + 1):
                continue

            for start_col in range(first_col, last_col, scan_width):

                start_col = int(start_col + (scan_width / 2.0) * (np.random.uniform() - 0.5))

                if np.random.uniform() > keep_prob:
                    continue

                end_col = start_col + window_width
            
                if (end_col > image.shape[1] + 1):
                    continue

                tile = image[start_row:end_row, start_col:end_col, :]
                # resize tile to 64 by 64
                tile = cv2.resize(tile, (64,64))

                bbox_pt1 = (start_col, start_row) #col, row
                bbox_pt2 = (end_col, end_row)
                imgs.append(tile)
                bboxes.append((bbox_pt1, bbox_pt2))

    features = extract_features_imgs(imgs, cspace="HLS", hog_channel="ALL")

    features = feature_scaler.transform(features)

    pred = clf.predict(features)

    pred = np.array(pred)
    all_boxes = np.array(bboxes)

    detected_boxes = all_boxes[(pred == 1)]

    print('len single scale all boxes: ', len(all_boxes))

    return detected_boxes, all_boxes

def detect_objects_multi_scale(image, clf, feature_scaler, min_size, max_size, step_size):
    #expect min_size, max_size, step_size to each be a 2-tuples (height, width) aka (rows,cols)
    widths = range(min_size[1], max_size[1], step_size[1])
    heights = range(min_size[0], max_size[0], step_size[0])
    L = min(len(widths), len(heights))

    widths = widths[0:L]
    heights = widths[0:L]

    detected_boxes = np.array([[[0,0], [0,0]]])
    all_boxes = np.array([ [[0,0], [0,0]] ])

    for width, height in zip(widths, heights):
        dboxes, aboxes = detect_objects(image, clf, feature_scaler, win_shape=(height, width), scan_shape=(height//7, width//7))

        detected_boxes = np.concatenate((detected_boxes, dboxes))
        all_boxes = np.concatenate((all_boxes, aboxes))
    
    detected_boxes = np.delete(detected_boxes, 0, 0)
    all_boxes = np.delete(all_boxes, 0, 0)

    print("number of scanned windows: ", len(all_boxes))
    print("detected boxes count: ", len(detected_boxes))

    return detected_boxes, all_boxes


def draw_bboxes(img, vehicle_boxes, all_boxes):
    draw_img = img.copy()
    print('drawing ', all_boxes.shape[0], ' scanning boxes')
    print('drawing ', vehicle_boxes.shape[0], ' vehicle detections')
    for bbox in all_boxes:
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (255,0,0))
    for bbox in vehicle_boxes:
        cv2.rectangle(draw_img, tuple(bbox[0]), tuple(bbox[1]), (0,255,0), thickness=6)
    return draw_img
            