import cv2
import numpy as np
from common import json
from common.json import GA_FORMAT
from common.video import Capture, Writer
from tqdm import tqdm

from display.group_activity import DisplayGroupActivity
from display.individual_activity import disp_individual_activity
from display.tracking import disp_tracking


def display(
    video_path,
    out_dir,
    individual_activity_json_path,
    group_activity_json_path,
    field,
    method=None,
):
    print("Prepareing video frames...")
    if method is None:
        methods = GA_FORMAT.keys()
    else:
        methods = [method]

    # out video file paths
    out_paths = [
        out_dir
        + "{}.mp4".format("individual_activity")
    ]
    for method in methods:
        out_paths.append(out_dir + "{}.mp4".format(method))

    # load datas
    individual_activity_datas = json.load(individual_activity_json_path)
    group_activity_datas = json.load(group_activity_json_path)

    display_group_activity = DisplayGroupActivity(group_activity_datas)

    # load video
    capture = Capture(video_path)
    print(video_path)

    # frames_lst = [[] for _ in range(len(out_paths))]
    cmb_img = combine_image(capture.read(), field)
    writer_lst = []
    for path in out_paths:
        if path.endswith("individual_activity.mp4"):
            size = capture.hw
        else:
            size = cmb_img.shape
        writer = Writer(path, capture.fps, size[1::-1])
        writer_lst.append(writer)

    # fields image array for plot all group activities
    group_activity_fields = [field.copy() for _ in range(len(methods))]

    # reset capture start position
    capture.set_pos_frame(0)

    for frame_num in tqdm(range(capture.frame_num)):
        # read frame
        frame = capture.read()

        # フレームごとにデータを取得する
        frame_individual_activity_datas = [
            data for data in individual_activity_datas if data["frame"] == frame_num
        ]

        # フレーム番号を表示
        cv2.putText(
            frame,
            "Frame:{}".format(frame_num + 1),
            (10, 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 0),
        )

        # copy raw field image
        field_tmp = field.copy()

        # draw tracking result
        frame = disp_tracking(frame_individual_activity_datas, frame)

        # draw individual activity
        field_tmp = disp_individual_activity(
            frame_individual_activity_datas, field_tmp, method
        )

        for i, method in enumerate(methods):
            group_activity_field = field_tmp.copy()
            group_activity_fields[i] = display_group_activity.disp(
                method, frame_num, group_activity_datas, group_activity_field
            )

        # write individual activity result
        writer_lst[0].write(combine_image(frame, field_tmp))

        # write group activity result
        for i in range(len(methods)):
            writer_lst[i + 1].write(combine_image(frame, group_activity_fields[i]))


def combine_image(frame, field):
    ratio = 1 - (field.shape[0] - frame.shape[0]) / field.shape[0]
    size = [int(field.shape[1] * ratio), int(field.shape[0] * ratio)]
    if frame.shape[0] != size[1]:
        # 丸め誤差が起きる
        size[1] = frame.shape[0]
    field = cv2.resize(field, size)
    frame = np.concatenate([frame, field], axis=1)

    return frame
