import numpy as np

class HeatMap(object):

    def __init__(self, shape):
        # initialize the heatmap with zeros (data type "int")
        self.shape = shape
        self.map = np.zeros(shape, dtype=np.int)

    def add_boxes(self, detection_boxes):
        # add some detection boxes to the heatmap
        # the heatmap is intended to be thresholded later to provide the
        # estimates of vehicle detections (to remove false positives from the output)
        # since the detections of false positives are expected to be less frequent over time
        for box in detection_boxes:
            self.map[box[0,1]:box[1,1], box[0,0]:box[1,0]] = self.map[box[0,1]:box[1,1], box[0,0]:box[1,0]] + 60
        self.map[:,:] = self.map[:,:] - 20 # 5 too small, 20 better but still too small? - was 50 - now 20
        self.map[self.map < 0] = 0
        self.map[self.map > 1024] = 1024