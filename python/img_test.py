from common import common, utils
from common.transform import Homography
from display import video
import cv2
import numpy as np


if __name__ == '__main__':
    # file path
    video_path = common.data_dir + 'basketball/basketball_alphapose.mp4'
    court_path = common.data_dir + 'basketball/court.png'

    # open video and image
    video = video.Video(video_path)
    court = cv2.imread(court_path)

    # utils.show_img(court)
    frame = video.read()
    utils.show_img(frame)

    p_video = np.float32([[210, 364], [1082, 362], [836, 488], [438, 489]])
    p_court = np.float32([[24, 24], [568, 24], [383, 232], [205, 232]])
    homo = Homography(p_video, p_court, court.shape)
    frame = homo.transform_image(frame)
    utils.show_img(frame)
