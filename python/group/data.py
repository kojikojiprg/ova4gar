from common import database
from person import data as pd
from group.group import Group


def make_database(person_db_path, group_db_path, homo):
    person_db = database.DataBase(person_db_path)
    group_db = database.DataBase(group_db_path)

    group = Group(homo)
    group_datas = []
    persons = pd.read_database(person_db, homo)
    frame_num = 0
    while True:
        person_datas = []
        for person in persons:
            person_data = person.get_data(frame_num, is_keypoints_numpy=False)
            if person_data is not None:
                person_datas.append(person_data)

        if len(person_datas) == 0:
            break

        group.append_calc(person_datas)

        data = group.get_data(frame_num)
        if data is not None:
            group_datas.append(data)

        frame_num += 1

    table = database.GROUP_TABLE
    group_db.drop_table(table.name)
    group_db.create_table(table.name, table.cols)
    group_db.insert_datas(
        table.name,
        list(table.cols.keys()),
        person_datas)
