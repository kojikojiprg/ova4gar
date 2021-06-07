from common import json
from common.json import GA_FORMAT
from display.video import Video
from display.tracking import disp_tracking
from display.individual_activity import disp_individual_activity
from display.group_activity import DisplayGroupActivity
import numpy as np
import cv2


def display(
        video_path,
        out_dir,
        individual_activity_json_path,
        group_activity_json_path,
        field,
        method=None,
        **karg):
    if method is None:
        methods = GA_FORMAT.keys()
    else:
        methods = [method]

    # out video file paths
    out_paths = [
        # out_dir + '{}.mp4'.format('tracking'),
        # out_dir + '{}.mp4'.format('individual_activity')
    ]
    for method in methods:
        if (
            'is_default_angle_range' in karg and
            not karg['is_default_angle_range'] and
            method == list(GA_FORMAT.keys())[0]
        ):
            # attention のとき、かつ視野角が異なるとき ファイル名に視野角を追加
            out_paths.append(
                out_dir + '{}_{}.mp4'.format(method, karg['angle_range'])
            )
        else:
            out_paths.append(
                out_dir + '{}.mp4'.format(method)
            )

    # load datas
    individual_activity_datas = json.load(individual_activity_json_path)
    group_activity_datas = json.load(group_activity_json_path)

    display_group_activity = DisplayGroupActivity(group_activity_datas)

    # load video
    video = Video(video_path)

    frames_lst = [[] for _ in range(len(out_paths))]
    group_activity_fields = [field.copy() for _ in range(len(methods))]
    for frame_num in range(video.frame_num):
        # read frame
        frame = video.read()

        # フレームごとにデータを取得する
        frame_individual_activity_datas = [
            data for data in individual_activity_datas if data['frame'] == frame_num]

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(frame_num + 1), (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        field_tmp = field.copy()

        # トラッキングの結果を表示
        frame = disp_tracking(frame_individual_activity_datas, frame)
        # 向きを表示
        field_tmp = disp_individual_activity(
            frame_individual_activity_datas, field_tmp, method)

        for i, method in enumerate(methods):
            group_activity_field = field_tmp.copy()
            group_activity_fields[i] = display_group_activity.disp(
                method, frame_num, group_activity_datas, group_activity_field)

        # append tracking result
        # frames_lst[0].append(frame)
        # frames_lst[1].append(combine_image(frame, field_tmp))
        for i in range(len(methods)):
            # frames_lst[i + 2].append(combine_image(frame,
            #                          group_activity_fields[i]))
            frames_lst[i].append(
                combine_image(frame, group_activity_fields[i]))

    for frames, out_path in zip(frames_lst, out_paths):
        print('writing video {} ...'.format(out_path))
        video.write(frames, out_path, frames[0].shape[1::-1])


def combine_image(frame, field):
    ratio = 1 - (field.shape[0] - frame.shape[0]) / field.shape[0]
    size = (int(field.shape[1] * ratio), int(field.shape[0] * ratio))
    field = cv2.resize(field, size)
    frame = np.concatenate([frame, field], axis=1)
    return frame
