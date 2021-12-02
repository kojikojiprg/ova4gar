import pickle

import numpy as np
from common.default import PASSING_DEFAULT
from common.functions import cos_similarity, gauss
from common.json import IA_FORMAT, START_IDX
from common.keypoint import body
from sklearn.svm import SVC


class PassingDetector:
    def __init__(
        self, model_path=None, C=PASSING_DEFAULT["C"], gamma=PASSING_DEFAULT["gamma"]
    ):
        if model_path is None:
            self.clf = SVC(C=C, gamma=gamma)
        else:
            with open(model_path, "rb") as f:
                self.clf = pickle.load(f)

    def predict(
        self,
        p1,
        p2,
        mu=PASSING_DEFAULT["gauss_mu"],
        sigma=PASSING_DEFAULT["gauss_sig"],
        wrist_mu=PASSING_DEFAULT["wrist_gauss_mu"],
        wrist_sig=PASSING_DEFAULT["wrist_gauss_sig"],
    ):
        p1_kps = p1[IA_FORMAT[START_IDX - 1]]
        p2_kps = p2[IA_FORMAT[START_IDX - 1]]
        p1_pos = p1[IA_FORMAT[START_IDX + 0]]
        p2_pos = p2[IA_FORMAT[START_IDX + 0]]
        p1_body = p1[IA_FORMAT[START_IDX + 2]]
        p2_body = p2[IA_FORMAT[START_IDX + 2]]
        p1_arm = p1[IA_FORMAT[START_IDX + 3]]
        p2_arm = p2[IA_FORMAT[START_IDX + 3]]

        if (
            p1_pos is None or
            p2_pos is None or
            p1_body is None or
            p2_body is None or
            p1_arm is None or
            p2_arm is None
        ):
            return None

        # calc distance
        norm = np.linalg.norm(np.array(p1_pos) - np.array(p2_pos), ord=2)
        distance = gauss(norm, mu=mu, sigma=sigma)

        # calc vector of each other
        p1_pos = np.array(p1_pos)
        p2_pos = np.array(p2_pos)
        p1p2 = p2_pos - p1_pos
        p2p1 = p1_pos - p2_pos

        p1p2_sim = cos_similarity(p1_body, p1p2)
        p2p1_sim = cos_similarity(p2_body, p2p1)
        body_direction = np.average([p1p2_sim, p2p1_sim])

        # calc arm average
        arm_ave = np.average([p1_arm, p2_arm])

        # calc wrist distance
        min_norm = np.inf
        wrist1 = (p1_kps[body["LWrist"]], p1_kps[body["RWrist"]])
        wrist2 = (p2_kps[body["LWrist"]], p2_kps[body["RWrist"]])
        for i in range(2):
            for j in range(2):
                norm = np.linalg.norm(np.array(wrist1[i]) - np.array(wrist2[j]), ord=2)
                if norm < min_norm:
                    min_norm = norm

        wrist_distance = gauss(min_norm, mu=wrist_mu, sigma=wrist_sig)

        feature = np.array([distance, body_direction, arm_ave, wrist_distance]).reshape(
            -1, 4
        )

        pred = self.clf.predict(feature)
        pred = pred.ravel()[0]  # extract value
        return pred
