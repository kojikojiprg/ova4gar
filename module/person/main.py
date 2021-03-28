from common import json
from person.person import Person


def main(tracking_json_path, person_json_path, homo):
    tracking_datas = json.load(tracking_json_path)

    persons = []
    person_datas = []
    for row in tracking_datas:
        person_id = row[database.TRACKING_TABLE.index('Person_ID')]
        frame_num = row[database.TRACKING_TABLE.index('Frame_No')]
        keypoints = row[database.TRACKING_TABLE.index('Keypoints')]
        vector = row[database.TRACKING_TABLE.index('Vector')]
        average = row[database.TRACKING_TABLE.index('Average')]

        if len(persons) == person_id:
            persons.append(Person(person_id, frame_num, homo))

        persons[person_id].calc_indicator(keypoints, vector, average)

        data = persons[person_id].get_data(frame_num)
        if data is not None:
            person_datas.append(data)

    table = database.PERSON_TABLE
    person_db.drop_table(table.name)
    person_db.create_table(table.name, table.cols)
    person_db.insert_datas(
        table.name,
        list(table.cols.keys()),
        person_datas)


def load_person_json(person_json_path, homo):
    person_data = json.load(person_json_path)

    # person data
    persons = []
    for data in person_data:
        person_id = data[database.PERSON_TABLE.index('Person_ID')]
        frame_num = data[database.PERSON_TABLE.index('Frame_No')]

        if len(persons) == person_id:
            persons.append(Person(person_id, frame_num, homo))

        persons[person_id].append_data(data)

    return persons
