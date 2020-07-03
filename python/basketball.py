import cv2
import numpy as np
from module import common, video, utils, keypoint, tracker


if __name__ == '__main__':
    video_path = common.data_dir + 'basketball/basketball_alphapose.mp4'
    out_path = common.out_dir + 'basketball/basketball_particle.mp4'
    court_path = common.data_dir + 'basketball/court.png'
    json_path = common.data_dir + 'basketball/keypoints.json'

    video = video.Video(video_path, out_path)
    court = cv2.imread(court_path)

    keypoints_frame = keypoint.Frame(json_path)

    # p_video = np.float32([[499, 364], [784, 363], [836, 488], [438, 489]])
    # p_court = np.float32([[205, 24], [383, 24], [383, 232], [205, 232]])
    person_id = 3
    tr = tracker.Tracker(keypoints_frame, ((15, 350), (1080, 632)))
    result = tr.track_person(person_id)

    frames = []
    for i, r in enumerate(result):
        frame = video.read()
        if r is not None:
            cv2.circle(frame, tuple(r), 7, (0, 0, 255), thickness=-1)
        frames.append(frame)
    video.write(frames)
