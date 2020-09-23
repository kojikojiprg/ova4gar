from common import database
from display.person import Person
from display.indicator import Indicator
from display.video import Video
import numpy as np
import cv2


def display(video_path, out_dir, tracking_db_path, indicator_db_path, field, homography):
    # out video file paths
    out_paths = [out_dir + '{}.mp4'.format(database.TRACKING_TABLE.name)]
    for table in database.INDICATOR_TABLES:
        out_paths.append(out_dir + '{}.mp4'.format(table.name))

    # connect database and read datas
    tracking_db = database.DataBase(tracking_db_path)
    indicator_db = database.DataBase(indicator_db_path)
    persons, indicators = read_sql(tracking_db, indicator_db)

    # load video
    video = Video(video_path)

    # copy field
    fields = [field.copy() for _ in range(len(database.INDICATOR_TABLES))]

    frames_lst = [[] for _ in range(len(out_paths))]
    for i in range(video.frame_num):
        # read frame
        frame = video.read()

        # フレーム番号を表示
        cv2.putText(frame, 'Frame:{}'.format(i + 1), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        # トラッキングを表示
        for person in persons:
            idx = i - person.start_frame_num
            if 0 <= idx and idx < len(person.keypoints_lst):
                point = person.keypoints_lst[idx]
                if point is not None:
                    point = point.get_middle('Hip')
                    cv2.circle(frame, tuple(point), 7, (0, 0, 255), thickness=-1)
                    cv2.putText(frame, str(person.id), tuple(point), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        frames_lst[0].append(frame)

        for indicator_idx, indicator in enumerate(indicators):
            frame_raw = frame.copy()
            if indicator_idx == 0:
                field_rslt = vector(i, indicator, fields[indicator_idx], homography)
                frame_rslt = combine_image(frame_raw, field_rslt)
            elif indicator_idx == 1:
                field_rslt = move_hand(i, indicator, fields[indicator_idx], homography)
                frame_rslt = combine_image(frame_raw, field_rslt)
            else:
                frames_lst = frame_raw

            frames_lst[indicator_idx + 1].append(frame_rslt)

    for frames, out_path in zip(frames_lst, out_paths):
        video.write(frames, out_path, frames[0].shape[1::-1])


def read_sql(tracking_db, indicator_db):
    persons = []

    # tracking data
    tracking_datas = tracking_db.select(database.TRACKING_TABLE.name)
    for tracking_data in tracking_datas:
        person_id = tracking_data[0]
        frame_num = tracking_data[1]

        if len(persons) == person_id:
            persons.append(Person(person_id, frame_num))

        persons[person_id].append(tracking_data)

    # indicator data
    indicators = []

    # read all data
    indicator_dict = {}
    for table in database.INDICATOR_TABLES:
        datas = indicator_db.select(table.name)
        indicator_dict[table.name] = datas

    # append datas
    for i, items in enumerate(indicator_dict.items()):
        table_name = items[0]
        indicator_datas = items[1]
        indicators.append(Indicator(table_name))

        for indicator_data in indicator_datas:
            for idx, key in enumerate(database.INDICATOR_TABLES[i].cols.keys()):
                if key == 'Frame_No':
                    break
            frame_num = indicator_data[idx]
            indicators[-1].append(frame_num, indicator_data)

    # make heatmaps
    for indicator in indicators:
        indicator.make_heatmap()

    return persons, indicators


def combine_image(frame, field):
    ratio = 1 - (frame.shape[0] - field.shape[0]) / frame.shape[0]
    size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
    frame = cv2.resize(frame, size)
    frame = np.concatenate([frame, field], axis=1)
    return frame


def vector(frame_num, indicator, field, homo):
    datas = indicator.indicator_lst[frame_num]
    for data in datas:
        if data[2] is None or data[3] is None:
            continue

        start = data[2]
        end = data[2] + data[3]
        start = homo.transform_point(start)
        end = homo.transform_point(end)
        color = data[4]

        cv2.arrowedLine(field, tuple(start), tuple(end), color, tipLength=1.5)

    return field


def move_hand(frame_num, indicator, field, homo):
    datas = indicator.indicator_lst[frame_num]
    for data in datas:
        if data[2] is None:
            continue

        point = data[2]
        point = homo.transform_point(point)
        color = data[3]
        cv2.circle(field, tuple(point), 7, color, thickness=-1)
    return field


def density(frame_num, indicator, field_raw, homo):
    pass
