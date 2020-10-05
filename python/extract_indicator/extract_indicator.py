from common import database
from extract_indicator.person import Person
from extract_indicator.frame import Frame
from extract_indicator import vector, move_hand, face_direction, density


def extract_indicator(tracking_db_path, indicator_db_path):
    tracking_db = database.DataBase(tracking_db_path)
    indicator_db = database.DataBase(indicator_db_path)

    persons, frames = read_sql(tracking_db)

    vector.calc_save(persons, indicator_db)
    move_hand.calc_save(persons, indicator_db)
    face_direction.calc_save(persons, indicator_db)
    density.calc_save(frames, indicator_db)


def read_sql(db):
    datas = db.select(database.TRACKING_TABLE.name)

    persons = []
    frames = []
    for row in datas:
        person_id = row[0]
        frame_num = row[1]
        keypoints = row[2]

        if len(persons) == person_id:
            persons.append(Person(person_id, frame_num))

        if len(frames) == frame_num:
            frames.append(Frame(frame_num))

        persons[person_id].append(row)
        frames[frame_num].append(keypoints)

    return persons, frames
