import os
import json


# tracking/tracking.py
# indivisual_activity/main.py
TRACKING_FORMAT = [
    'label',
    'frame',
    'keypoints',
    'vector',
    'average',
]

# indivisual_activity/indivisual_activity.py
# indivisual_activity/indicator.py
# group_activity/indicator.py
# display/tracking.py
# display/indivisual_activity.py
IA_FORMAT = [
    'label',
    'frame',
    'keypoints',
    'position',
    'face_vector',
    'body_vector',
    'arm',
]

ATTENTION_FORMAT = [
    'frame',
    'point',
    'count',
]

PASSING_FORMAT = [
    'frame',
    'point',
    'persons',
    'likelihood',
]

# group_activity/group_activity.py
# group_activity/indicator.py
# display/display.py
# display/group_activity.py
GA_FORMAT = {
    'attention': ATTENTION_FORMAT,
    'passing': PASSING_FORMAT,
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
