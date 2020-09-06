from module import common, video, utils
from transform import Homography
import cv2
import numpy as np


if __name__ == '__main__':
    # file path
    name = 'tenis'
    video_path = common.data_dir + '{0}/{0}_alphapose.mp4'.format(name)
    court_path = common.data_dir + '{}/court.png'.format(name)

    # open video and image
    video = video.Video(video_path)
    court = cv2.imread(court_path)

    utils.show_img(court)
    frame = video.read()
    utils.show_img(frame)

    p_video = np.float32([[381, 201], [897, 201], [1104, 567], [171, 567]])
    p_court = np.float32([[27, 24], [160, 24], [160, 238], [27, 238]])
    homo = Homography(p_video, p_court, court.shape)
    frame = homo.transform_image(frame)
    utils.show_img(frame)
