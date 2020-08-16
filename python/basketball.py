import cv2
import numpy as np
from module import common, video, keypoint, tracker, transform
from heatmap import Heatmap

# パラメータ
MODE = [
    'none',
    'speed',
    'move-hand',
    'vector',
]
MODE_NUM = 0

SHOW_PARTICLE = False
SHOW_POINT = True
SHOW_ID = True

if __name__ == '__main__':
    # file path
    video_path = common.data_dir + 'basketball/basketball_alphapose.mp4'
    out_path = common.out_dir + 'basketball/basketball_particle_{}.mp4'.format(MODE[MODE_NUM])
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
    tr = tracker.Tracker(keypoints_frame[0])

    # heatmap
    #heatmaps = [Heatmap(p, homo) for p in persons]

    frames = []
    for i in range(video.frame_num):
        # read frame
        frame = video.read()

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(i + 1), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        keypoints_lst = keypoints_frame[i]
        persons = tr.track(i, keypoints_lst)

        for j, person in enumerate(persons):
            point = person.keypoints_lst[-1]
            particles = person.particles_lst[-1]
            #heatmap = heatmaps[j]

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
                # スピードのヒートマップを表示
                if heatmap.verocity_map[i] is not None:
                    now = heatmap.verocity_map[i][0]
                    nxt = heatmap.verocity_map[i][1]
                    color = heatmap.verocity_map[i][2]
                    cv2.line(court, now, nxt, color, 3)
            elif MODE_NUM == 2:
                # 手の動きのヒートマップを表示
                if heatmap.move_hand_map[i] is not None:
                    now = heatmap.move_hand_map[i][0]
                    now = homo.transform_point(now)
                    color = heatmap.move_hand_map[i][1]
                    cv2.circle(court, now, 7, color, thickness=-1)
            elif MODE_NUM == 3:
                # ベクトルを表示
                if heatmap.vector_map[i] is not None:
                    start = heatmap.vector_map[i][0]
                    end = heatmap.vector_map[i][1]
                    color = heatmap.vector_map[i][2]
                    cv2.arrowedLine(court, start, end, color, tipLength=1.5)

        # 画像を合成
        ratio = 1 - (frame.shape[0] - court.shape[0]) / frame.shape[0]
        size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
        frame = cv2.resize(frame, size)
        frame = np.concatenate([frame, court], axis=1)

        frames.append(frame)

    video.write(frames, out_path, frame.shape[1::-1])
