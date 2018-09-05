import numpy as np

class HeatMap(object):

    def __init__(self, shape):
        self.shape = shape
        self.map = np.zeros(shape, dtype=np.uint8)

    def add_boxes(self, detection_boxes):
        for box in detection_boxes:
            self.map[box[0,1]:box[1,1], box[0,0]:box[1,0]] = self.map[box[0,1]:box[1,1], box[0,0]:box[1,0]] + 1