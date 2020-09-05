import cv2
import numpy as np
from module import common, video, keypoint, tracker, transform

# パラメータ
MODE = [
    'none',         # 0
    'vector',
    'move-hand',
    'population',   # 3
]
MODE_NUM = 3

SHOW_PARTICLE = True
SHOW_POINT = True
SHOW_ID = True

if __name__ == '__main__':
    # file path
    name = 'basketball'
    video_path = common.data_dir + '{0}/{0}_alphapose.mp4'.format(name)
    out_path = common.out_dir + '{0}/{0}_{1}.mp4'.format(name, MODE[MODE_NUM])
    court_path = common.data_dir + '{}/court.png'.format(name)
    json_path = common.data_dir + '{}/keypoints.json'.format(name)

    # open video and image
    video = video.Video(video_path)
    court_raw = cv2.imread(court_path)
    court = court_raw.copy()

    # homography
    p_video = np.float32([[210, 364], [1082, 362], [836, 488], [438, 489]])
    p_court = np.float32([[24, 24], [568, 24], [383, 232], [205, 232]])
    homo = transform.Homography(p_video, p_court, court.shape)

    # keypoints
    keypoints_frame = keypoint.Frame(json_path)

    # tracking
    tr = tracker.Tracker(keypoints_frame[0], homo)

    frames = []
    for i in range(video.frame_num):
        # read frame
        frame = video.read()

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(i + 1), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        keypoints_lst = keypoints_frame[i]
        persons, populations = tr.track(i, keypoints_lst)

        for j, person in enumerate(persons):
            point = person.keypoints_lst[-1]
            particles = person.particles_lst[-1]

            # パーティクルを表示
            if SHOW_PARTICLE and particles is not None:
                for par in particles:
                    cv2.circle(frame, (int(par[0]), int(par[1])), 2, (0, 255, 0), thickness=-1)

            # ポイントを表示
            if SHOW_POINT and point is not None:
                tmp = point.get_middle('Hip')
                cv2.circle(frame, tuple(tmp), 7, (0, 0, 255), thickness=-1)

            # IDを表示
            if SHOW_ID and point is not None:
                tmp = point.get_middle('Hip')
                cv2.putText(frame, str(j), tuple(tmp), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            if MODE_NUM == 1:
                # ベクトルを表示
                vector_map = person.vector_map[-1]
                if vector_map is not None:
                    start = vector_map[0]
                    end = vector_map[1]
                    start = homo.transform_point(start)
                    end = homo.transform_point(end)
                    color = vector_map[2]
                    cv2.arrowedLine(court, tuple(start), tuple(end), color, tipLength=1.5)
            elif MODE_NUM == 2:
                # 手の動きのヒートマップを表示
                move_hand_map = person.move_hand_map[-1]
                if move_hand_map is not None:
                    now = move_hand_map[0]
                    now = homo.transform_point(now)
                    color = move_hand_map[1]
                    cv2.circle(court, tuple(now), 7, color, thickness=-1)
            elif MODE_NUM == 3:
                # 毎回コート画像を読み込む
                court = court_raw.copy()
                # 人口密度を表示
                population = populations[-1]
                for item in population:
                    p1 = item[0]
                    p2 = item[1]
                    color = item[2]
                    cv2.rectangle(court, p1, p2, color, thickness=-1)

        # 画像を合成
        ratio = 1 - (frame.shape[0] - court.shape[0]) / frame.shape[0]
        size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
        frame = cv2.resize(frame, size)
        frame = np.concatenate([frame, court], axis=1)

        frames.append(frame)

    video.write(frames, out_path, frame.shape[1::-1])
