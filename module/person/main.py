from common import json, TRACKING_FORMAT
from person.person import Person


def main(tracking_json_path, person_json_path, homo):
    tracking_datas = json.load(tracking_json_path)

    persons = []
    person_datas = []
    for item in tracking_datas.items():
        person_id = item[TRACKING_FORMAT[0]]
        frame_num = item[TRACKING_FORMAT[1]]
        keypoints = item[TRACKING_FORMAT[2]]
        vector = item[TRACKING_FORMAT[3]]
        average = item[TRACKING_FORMAT[4]]

        if len(persons) == person_id:
            persons.append(Person(person_id, frame_num, homo))

        persons[person_id].calc_indicator(keypoints, vector, average)

        data = persons[person_id].to_json(frame_num)
        if data is not None:
            person_datas.append(data)

    json.dump(person_datas)
