from common import database
from person.person import Person


def make_database(tracking_db_path, person_db_path, homo):
    tracking_db = database.DataBase(tracking_db_path)
    person_db = database.DataBase(person_db_path)

    tracking_datas = tracking_db.select(database.TRACKING_TABLE.name)

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

        persons[person_id].append_calc(keypoints, vector, average)

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


def read_database(person_db_path, homo):
    person_db = database.DataBase(person_db_path)
    person_datas = person_db.select(database.PERSON_TABLE.name)

    # person data
    persons = []
    for data in person_datas:
        person_id = data[database.PERSON_TABLE.index('Person_ID')]
        frame_num = data[database.PERSON_TABLE.index('Frame_No')]

        if len(persons) == person_id:
            persons.append(Person(person_id, frame_num, homo))

        persons[person_id].append_data(data)

    return persons
