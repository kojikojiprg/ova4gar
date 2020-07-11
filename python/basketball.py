import cv2
import numpy as np
from module import common, video, utils, keypoint, tracker, transform


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
    person_id = 8

    # tracking
    tr = tracker.Tracker(keypoints_frame)
    result, particles = tr.track_person(person_id)

    frames = []
    for i, z in enumerate(zip(result, particles)):
        r = z[0]    # result point
        p = z[1]    # particles

        # read frame
        frame = video.read()

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(i + 1), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        # パーティクルを表示
        for par in p:
            cv2.circle(frame, (int(par[0]), int(par[1])), 2, (0, 255, 0), thickness=-1)

        # ポイントを表示
        if r is not None:
            # add point on a frame
            cv2.circle(frame, tuple(r), 7, (0, 0, 255), thickness=-1)

            # homography convert
            homo_p = homo.transform_point(r)
            cv2.circle(court, tuple(homo_p), 7, (0, 0, 255), thickness=-1)

        # 画像を合成
        ratio = 1 - (frame.shape[0] - court.shape[0]) / frame.shape[0]
        size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
        frame = cv2.resize(frame, size)
        frame = np.concatenate([frame, court], axis=1)
        #utils.show_img(frame)

        frames.append(frame)
    print(frame.shape)
    video.write(frames, out_path, frame.shape[1::-1])
