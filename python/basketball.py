import cv2
import numpy as np
from module import common, video, transform, utils, keypoint


if __name__ == '__main__':
    video_path = common.data_dir + 'basketball/basketball.mp4'
    court_path = common.data_dir + 'basketball/court.png'
    json_path = common.data_dir + 'basketball/keypoints.json'

    reader = video.Reader(video_path)
    court = cv2.imread(court_path)

    keypoint_frame = keypoint.Frame(json_path)

    for keypoints_lst in keypoint_frame:
        frame = reader.read()
        keypoints = keypoints_lst[10]
        cv2.circle(frame, keypoints.get('LAnkle')[:2], 7, (0, 0, 255), thickness=-1)
        cv2.circle(frame, keypoints.get('RAnkle')[:2], 7, (255, 0, 0), thickness=-1)
        utils.show_img(frame)

    """
    p_video = np.float32([[499, 364], [784, 363], [836, 488], [438, 489]])
    p_img = np.float32([[205, 24], [383, 24], [383, 232], [205, 232]])
    frame_transform = transform.homography(frame, p_video, p_img)
    out = np.concatenate((frame_transform, img), axis=1)
    utils.show_img(out, is_save=False, save_name=common.out_dir + 'basketball/out.png')
    """
