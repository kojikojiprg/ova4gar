from common import common, keypoint, database


def extract_indicator(tracking_db_path, analysis_db_path):
    analysis_db = database.DataBase(analysis_db_path)

    persons, frames = keypoint.read_sql(tracking_db_path)
    print(persons[0])
