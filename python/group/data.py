from common import database
from person import data as pd
from group.group import Group


def make_database(person_db_path, group_db_path, homo):
    persons = pd.read_database(person_db_path, homo)

    group_db = database.DataBase(group_db_path)

    group = Group(homo)

    for table in database.GROUP_TABLE_LIST:
        indicator_datas = []
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
            group.append_calc(table.name, frame_num, person_datas)

            # データベース用のリストに追加
            indicator_datas += group.get_data(table.name, frame_num)

            frame_num += 1

        group_db.drop_table(table.name)
        group_db.create_table(table.name, table.cols)
        group_db.insert_datas(
            table.name,
            list(table.cols.keys()),
            indicator_datas)


def read_database(group_db_path, homo):
    group_db = database.DataBase(group_db_path)

    group = Group(homo)

    for table in database.GROUP_TABLE_LIST:
        datas = group_db.select(table.name)
        group.append_data(table.name, datas)

    group.make_heatmap()

    return group
