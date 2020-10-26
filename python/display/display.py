from common import database
import person.data as pd
import group.data as gd
from display.video import Video
import numpy as np
import cv2


def display(video_path, out_dir, person_db_path, group_db_path, field, homo):
    # out video file paths
    out_paths = [
        out_dir + '{}.mp4'.format(database.TRACKING_TABLE.name),
        out_dir + '{}.mp4'.format(database.PERSON_TABLE.name)
    ]
    for table in database.GROUP_TABLE_LIST:
        out_paths.append(
            out_dir + '{}.mp4'.format(table.name)
        )

    # connect database and read datas
    persons = pd.read_database(person_db_path, homo)
    group = gd.read_database(group_db_path, homo)

    # load video
    video = Video(video_path)

    frames_lst = [[] for _ in range(len(out_paths))]
    group_fields = [field.copy() for _ in range(len(database.GROUP_TABLE_LIST))]
    for frame_num in range(video.frame_num):
        # read frame
        frame = video.read()

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(frame_num + 1), (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        field_tmp = field.copy()
        for person in persons:
            # トラッキングを表示
            frame = person.display_tracking(frame_num, frame)
            # 向きを表示
            field_tmp = person.display_vector(frame_num, field_tmp)

        for i, table in enumerate(database.GROUP_TABLE_LIST):
            group_field = field.copy()
            group_fields[i] = group.display(table.name, frame_num, group_field)

        # append tracking result
        frames_lst[0].append(frame)
        frames_lst[1].append(combine_image(frame, field_tmp))
        for i in range(len(database.GROUP_TABLE_LIST)):
            frames_lst[i + 2].append(combine_image(frame, group_fields[i]))

    for frames, out_path in zip(frames_lst, out_paths):
        video.write(frames, out_path, frames[0].shape[1::-1])


def combine_image(frame, field):
    ratio = 1 - (frame.shape[0] - field.shape[0]) / frame.shape[0]
    size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
    frame = cv2.resize(frame, size)
    frame = np.concatenate([frame, field], axis=1)
    return frame
