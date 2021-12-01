from common import common, utils
from common.transform import Homography
from display import video
import cv2

if __name__ == '__main__':
    # file path
    room_num = '08'
    video_path = common.root + '/video_raw/08_05_Survey_20210915_070700_01.mp4'
    field_path = common.data_dir + '{}/field.png'.format(room_num)
    print(video_path)
    print(field_path)

    # open video and image
    video = video.Video(video_path)
    frame = video.read()
    utils.show_img(frame)

    field = cv2.imread(field_path)
    utils.show_img(field)

    p_video = common.homo[room_num][0]
    p_field = common.homo[room_num][1]
    homo = Homography(p_video, p_field, field.shape)
    frame = homo.transform_image(frame)
    utils.show_img(frame)
