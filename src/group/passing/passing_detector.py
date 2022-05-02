from types import SimpleNamespace

import numpy as np
import torch
import yaml
from group.passing.lstm_model import LSTMModel
from individual.individual import Individual
from utility.functions import cos_similarity, gauss


class PassingDetector:
    def __init__(self, cfg_path: str, defs: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load config of model
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self._model = LSTMModel(**self.cfg)
        self._seq_len = self.cfg["seq_len"]

        param = torch.load(self.cfg["pretrained_path"])
        self._model.load_state_dict(param)
        self._model.to(self.device)

        # load defaults
        self._mu = defs["gauss_mu"]
        self._sigma = defs["gauss_sig"]
        self._wrist_mu = defs["wrist_gauss_mu"]
        self._wrist_sig = defs["wrist_gauss_sig"]

    def __del__(self):
        del self._model

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    @staticmethod
    def _get_indicators(ind: Individual, frame_num: int):
        pos = ind.get_indicator("position", frame_num)
        body = ind.get_indicator("body", frame_num)
        arm = ind.get_indicator("arm", frame_num)
        wrist = (
            ind.get_keypoints("LWrist", frame_num),
            ind.get_keypoints("RWrist", frame_num),
        )
        ret = SimpleNamespace(**{"pos": pos, "body": body, "arm": arm, "wrist": wrist})
        return ret

    def extract_feature(
        self, ind1: Individual, ind2: Individual, que: list, frame_num: int
    ):
        # get indicator
        ind1_data = self._get_indicators(ind1, frame_num)
        ind2_data = self._get_indicators(ind2, frame_num)

        if None in ind1_data.__dict__.values() or None in ind2_data.__dict__.values():
            return None

        # calc distance of position
        p1_pos = np.array(ind1_data.pos)
        p2_pos = np.array(ind2_data.pos)

        norm = np.linalg.norm(p1_pos - p2_pos, ord=2)
        distance = gauss(norm, mu=self._mu, sigma=self._sigma)

        p1p2 = p2_pos - p1_pos
        p2p1 = p1_pos - p2_pos

        p1p2_sim = cos_similarity(ind1_data.body, p1p2)
        p2p1_sim = cos_similarity(ind2_data.body, p2p1)
        body_distance = (np.average([p1p2_sim, p2p1_sim]) + 1) / 2

        # calc arm average
        arm_ave = np.average([ind1_data.arm, ind2_data.arm])

        # calc wrist distance
        min_norm = np.inf
        for i in range(2):
            for j in range(2):
                norm = np.linalg.norm(
                    np.array(ind1_data.wrist[i]) - np.array(ind2_data.wrist[j]), ord=2
                )
                min_norm = float(norm) if norm < min_norm else min_norm

        wrist_distance = gauss(min_norm, mu=self._wrist_mu, sigma=self._wrist_sig)

        # concatnate to feature
        feature = [distance, body_distance, arm_ave, wrist_distance]
        que.append(feature)

        return que[-self._seq_len :]

    def predict(self, features):
        with torch.no_grad():
            features = torch.Tensor(np.array([features])).float().to(self.device)
            pred = self._model(features)
            pred = pred.max(1)[1]
            pred = pred.cpu().numpy()[0]

        return pred
