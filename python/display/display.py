from common import database
import person as ps
from display.video import Video
import numpy as np
import cv2


def display(video_path, out_dir, person_db_path, field, homo):
    # out video file paths
    out_paths = [
        out_dir + '{}.mp4'.format(database.TRACKING_TABLE.name),
        out_dir + '{}.mp4'.format(database.PERSON_TABLE.name)
    ]

    # connect database and read datas
    person_db = database.DataBase(person_db_path)
    persons = ps.data.read_database(person_db, homo)

    # load video
    video = Video(video_path)

    frames_lst = [[] for _ in range(len(out_paths))]
    for i in range(video.frame_num):
        # read frame
        frame = video.read()

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(i + 1), (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        field_tmp = field.copy()
        for person in persons:
            # トラッキングを表示
            frame = person.display_tracking(i, frame)
            # 向きを表示
            field_tmp = person.display_vector(i, field_tmp)

        # append tracking result
        frames_lst[0].append(frame)
        frames_lst[1].append(combine_image(frame, field_tmp))

    for frames, out_path in zip(frames_lst, out_paths):
        video.write(frames, out_path, frames[0].shape[1::-1])


def combine_image(frame, field):
    ratio = 1 - (frame.shape[0] - field.shape[0]) / frame.shape[0]
    size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
    frame = cv2.resize(frame, size)
    frame = np.concatenate([frame, field], axis=1)
    return frame
