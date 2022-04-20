import cv2

from common import common, transform, utils, video
from common.transform import Homography

if __name__ == "__main__":
    # file path
    room_num = "08"
    date = "20210915"
    video_path = common.root + f"/video/{room_num}/{date}/01.mp4"
    field_path = common.data_dir + "/{}/field.png".format(room_num)
    print(video_path)
    print(field_path)

    # open video and image
    video = video.Capture(video_path)
    frame = video.read()
    utils.show_img(frame)

    field = cv2.imread(field_path)
    utils.show_img(field)

    p_video = transform.homo[room_num][0]
    p_field = transform.homo[room_num][1]
    homo = Homography(p_video, p_field, field.shape)
    frame = homo.transform_image(frame)
    utils.show_img(frame)
