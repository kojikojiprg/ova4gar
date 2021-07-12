import os
import json
import numpy as np


# tracking/tracking.py
# individual_activity/main.py
TRACKING_FORMAT = [
    'label',
    'frame',
    'keypoints',
    'vector',
    'average',
]

# individual_activity/individual_activity.py
# individual_activity/indicator.py
# group_activity/indicator.py
# display/tracking.py
# display/individual_activity.py
IA_FORMAT = [
    'label',
    'frame',
    'tracking_position',
    'position',
    'face_vector',
    'body_vector',
    'arm',
]


# group_activity/group_activity.py
# group_activity/indicator.py
# display/display.py
# display/group_activity.py
ATTENTION_FORMAT = [
    'frame',
    'point',
    'person_points',
    'count',
]

PASSING_FORMAT = [
    'frame',
    'persons',
    'points',
    'precision',
]

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
        json.dump(data, f, cls=MyEncoder)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
