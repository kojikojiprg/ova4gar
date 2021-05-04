from common import json
from common.json import GROUP_FORMAT
from display.video import Video
from display.tracking import disp_tracking
from display.person import disp_person
from display.group import DisplayGroup
import numpy as np
import cv2


def display(video_path, out_dir, person_json_path, group_json_path, field, method=None, **karg):
    if method is None:
        methods = GROUP_FORMAT.keys()
    else:
        methods = [method]

    # out video file paths
    out_paths = [
        out_dir + '{}.mp4'.format('tracking'),
        out_dir + '{}.mp4'.format('person')
    ]
    for method in methods:
        if not karg['is_default_angle_range'] and method == 'attention':
            # attention のとき、かつ視野角が異なるとき ファイル名に視野角を追加
            out_paths.append(
                out_dir + '{}_{}.mp4'.format(method, karg['angle_range'])
            )
        else:
            out_paths.append(
                out_dir + '{}.mp4'.format(method)
            )

    # load datas
    person_datas = json.load(person_json_path)
    group_datas = json.load(group_json_path)

    display_group = DisplayGroup(group_datas)

    # load video
    video = Video(video_path)

    frames_lst = [[] for _ in range(len(out_paths))]
    group_fields = [field.copy() for _ in range(len(methods))]
    for frame_num in range(video.frame_num):
        # read frame
        frame = video.read()

        # フレームごとにデータを取得する
        frame_person_datas = [
            data for data in person_datas if data['image_id'] == frame_num]

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(frame_num + 1), (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        field_tmp = field.copy()

        # トラッキングの結果を表示
        frame = disp_tracking(frame_person_datas, frame)
        # 向きを表示
        field_tmp = disp_person(frame_person_datas, field_tmp)

        for i, method in enumerate(methods):
            group_field = field_tmp.copy()
            group_fields[i] = display_group.disp(
                method, frame_num, group_datas, group_field)

        # append tracking result
        frames_lst[0].append(frame)
        frames_lst[1].append(combine_image(frame, field_tmp))
        for i in range(len(methods)):
            frames_lst[i + 2].append(combine_image(frame, group_fields[i]))

    print('writing videos into {} ...'.format(out_dir))
    for frames, out_path in zip(frames_lst, out_paths):
        video.write(frames, out_path, frames[0].shape[1::-1])


def combine_image(frame, field):
    ratio = 1 - (field.shape[0] - frame.shape[0]) / field.shape[0]
    size = (int(field.shape[1] * ratio), int(field.shape[0] * ratio))
    field = cv2.resize(field, size)
    frame = np.concatenate([frame, field], axis=1)
    return frame
