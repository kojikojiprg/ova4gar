import numpy as np
import cv2


class Homography:
    def __init__(self, p_src, p_dst, dst_size):
        self.M = cv2.getPerspectiveTransform(p_src, p_dst)
        self.size = dst_size

    def transform_image(self, src):
        return cv2.warpPerspective(src, self.M, self.size)

    def transform_point(self, point):
        point = np.append(point, 1)
        result = np.dot(self.M, point)
        return np.array([
            result[0] / result[2],
            result[1] / result[2],
        ])
