import os
import json


# tracking/tracking.py
# person/main.py
TRACKING_FORMAT = [
    'person_id',
    'image_id',
    'keypoints',
    'vector',
    'average',
]

# person/person.py
# person/indicator.py
# group/indicator.py
PERSON_FORMAT = [
    'person_id',
    'image_id',
    'keypoints',
    'position',
    'face_vector',
    'body_vector',
    'wrist',
]

DENSITY_FORMAT = [
    'image_id',
    'cluster',
    'count',
]

ATTENTION_FORMAT = [
    'image_id',
    'point',
    'count',
]

# group/indicator.py
GROUP_FORMAT = {
    'density': DENSITY_FORMAT,
    'attention': ATTENTION_FORMAT,
}


def load(json_path):
    data = {}
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def dump(data, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(data, f)
