from common import json
from tqdm import tqdm


def main(
        individual_activity_json_path,
        group_activity_json_path,
        field,
        **karg):
    print('Running situation recognition...')
    individual_activity_datas = json.load(individual_activity_json_path)
    group_activity_datas = json.load(group_activity_json_path)

    # jsonフォーマットを生成して書き込み
    json.dump(group_activity_datas, group_activity_json_path)
