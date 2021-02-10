from common import common, utils
from common.transform import Homography
from display import video
import cv2


if __name__ == '__main__':
    # file path
    name = 'or'
    video_path = common.data_dir + '{0}/sugukesu.png'.format(name)
    court_path = common.data_dir + '{}/field.png'.format(name)

    # open video and image
    frame = cv2.imread(video_path)
    utils.show_img(frame)

    court = cv2.imread(court_path)
    utils.show_img(court)

    p_video = common.homo[name][0]
    p_court = common.homo[name][1]
    homo = Homography(p_video, p_court, court.shape)
    frame = homo.transform_image(frame)
    utils.show_img(frame)
