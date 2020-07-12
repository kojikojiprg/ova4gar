import cv2
import numpy as np
from module import common, video, utils, keypoint, tracker, transform
from heatmap import Heatmap


if __name__ == '__main__':
    # file path
    video_path = common.data_dir + 'basketball/basketball_alphapose.mp4'
    out_path = common.out_dir + 'basketball/basketball_particle.mp4'
    court_path = common.data_dir + 'basketball/court.png'
    json_path = common.data_dir + 'basketball/keypoints.json'

    # open video and image
    video = video.Video(video_path)
    court = cv2.imread(court_path)

    # homography
    p_video = np.float32([[499, 364], [784, 363], [836, 488], [438, 489]])
    p_court = np.float32([[205, 24], [383, 24], [383, 232], [205, 232]])
    homo = transform.Homography(p_video, p_court, court.shape)

    # keypoints
    keypoints_frame = keypoint.Frame(json_path)

    # tracking
    person_id = 8
    tr = tracker.Tracker(keypoints_frame)
    points, particles, speeds = tr.track_person(person_id)
    speed_max = max(speeds)
    speed_min = min(speeds)
    speed_heatmap = Heatmap(speed_min, speed_max)

    frames = []
    prepoint = None
    for i, rslt in enumerate(zip(points, particles, speeds)):
        point = rslt[0]         # result point
        particles = rslt[1]     # particles
        speed = rslt[2]         # speed heatmap

        # read frame
        frame = video.read()

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(i + 1), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        # パーティクルを表示
        for par in particles:
            cv2.circle(frame, (int(par[0]), int(par[1])), 2, (0, 255, 0), thickness=-1)

        # ポイントを表示
        if point is not None:
            # add point on a frame
            cv2.circle(frame, tuple(point), 7, (0, 0, 255), thickness=-1)

        # スピードのヒートマップを表示
        if i > 1:
            p = homo.transform_point(point)
            pre = homo.transform_point(prepoint)
            speed = speed_heatmap.calc(speed)
            cv2.line(court, tuple(pre), tuple(p), tuple(speed), 3)
        prepoint = point

        # 画像を合成
        ratio = 1 - (frame.shape[0] - court.shape[0]) / frame.shape[0]
        size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
        frame = cv2.resize(frame, size)
        frame = np.concatenate([frame, court], axis=1)
        #utils.show_img(frame)

        frames.append(frame)

    video.write(frames, out_path, frame.shape[1::-1])
