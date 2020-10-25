from common import database
from person import data as pd
from group.group import Group


def make_database(person_db_path, group_db_path, homo):
    person_db = database.DataBase(person_db_path)
    group_db = database.DataBase(group_db_path)

    persons = pd.read_database(person_db, homo)

    group = Group(homo)
    group_datas = []
    frame_num = 0
    while True:
        # フレームごとに全ての人のデータを取得する
        person_datas = []
        for person in persons:
            person_data = person.get_data(frame_num, is_keypoints_numpy=False)
            if person_data is not None:
                person_datas.append(person_data)

        if len(person_datas) == 0:
            # データがなくなったら終了
            break

        # グループに追加して指標を計算
        group.append_calc(person_datas)

        # データベース用のリストに追加
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


def read_database(group_db_path, homo):
    group_db = database.DataBase(group_db_path)
    group_datas = group_db.select(database.PERSON_TABLE.name)

    # person data
    group = Group(homo)
    for data in group_datas:
        group.append_data(data)

    return group
