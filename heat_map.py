import numpy as np

class HeatMap(object):

    def __init__(self, shape):
        self.shape = shape
        self.map = np.zeros(shape, dtype=np.int)

    def add_boxes(self, detection_boxes):
        for box in detection_boxes:
            self.map[box[0,1]:box[1,1], box[0,0]:box[1,0]] = self.map[box[0,1]:box[1,1], box[0,0]:box[1,0]] + 100
        self.map[:,:] = self.map[:,:] - 20 # 5 too small, 20 better but still too small? - was 50 - now 20
        self.map[self.map < 0] = 0
        self.map[self.map > 1024] = 1024