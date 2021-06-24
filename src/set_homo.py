from common import common, utils
from common.transform import Homography
from display import video
import cv2

"""
if __name__ == '__main__':
    # file path
    room_num = '09'
    date = '20210304'
    video_path = common.data_dir + '{0}/{1}/pass1.png'.format(room_num, date)
    field_path = common.data_dir + 'field.png'

    # open video and image
    frame = cv2.imread(video_path)
    utils.show_img(frame)

    court = cv2.imread(field_path)
    utils.show_img(court)

    p_video = common.homo[room_num][0]
    p_court = common.homo[room_num][1]
    homo = Homography(p_video, p_court, court.shape)
    frame = homo.transform_image(frame)
    utils.show_img(frame)
"""

if __name__ == '__main__':
    # file path
    room_num = '09'
    date = '20210304'
    name = 'gaze2-1'
    video_path = common.data_dir + '{0}/{1}/{2}/video/{2}.mp4'.format(room_num, date, name)
    field_path = common.data_dir + 'field.png'
    print(video_path)

    # open video and image
    video = video.Video(video_path)
    frame = video.read()
    utils.show_img(frame)

    court = cv2.imread(field_path)
    utils.show_img(court)

    p_video = common.homo[room_num][0]
    p_court = common.homo[room_num][1]
    homo = Homography(p_video, p_court, court.shape)
    frame = homo.transform_image(frame)
    utils.show_img(frame)
