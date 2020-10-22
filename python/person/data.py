from common import database
from person.person import Person


def make_database(tracking_db_path, person_db_path, homo):
    tracking_db = database.DataBase(tracking_db_path)
    person_db = database.DataBase(person_db_path)

    tracking_datas = tracking_db.select(database.TRACKING_TABLE.name)

    models = []
    person_datas = []
    for row in tracking_datas:
        person_id = row[0]
        frame_num = row[1]
        keypoints = row[2]

        if len(models) == person_id:
            models.append(Person(person_id, frame_num, homo))

        models[person_id].append_calc(keypoints)

        data = models[person_id].get_data(frame_num)
        if data is not None:
            person_datas.append(data)

    table = database.PERSON_TABLE
    person_db.drop_table(table.name)
    person_db.create_table(table.name, table.cols)
    person_db.insert_datas(
        table.name,
        list(table.cols.keys()),
        person_datas)


def read_database(person_db, homo):
    persons = []

    # person data
    person_datas = person_db.select(database.PERSON_TABLE.name)
    for data in person_datas:
        person_id = data[0]
        frame_num = data[1]

        if len(persons) == person_id:
            persons.append(Person(person_id, frame_num, homo))

        persons[person_id].append_data(data)

    return persons
