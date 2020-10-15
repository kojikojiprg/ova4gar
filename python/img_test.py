from common import common, utils
from common.transform import Homography
from display import video
import cv2


if __name__ == '__main__':
    # file path
    name = 'basketball'
    video_path = common.data_dir + '{0}/{0}_alphapose.mp4'.format(name)
    court_path = common.data_dir + '{}/court.png'.format(name)

    # open video and image
    video = video.Video(video_path)
    court = cv2.imread(court_path)

    utils.show_img(court)
    frame = video.read()
    utils.show_img(frame)

    p_video = common.homo[name][0]
    p_court = common.homo[name][1]
    homo = Homography(p_video, p_court, court.shape)
    frame = homo.transform_image(frame)
    utils.show_img(frame)
