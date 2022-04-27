from types import SimpleNamespace

import numpy as np
import torch
import yaml
from keypoint.keypoint import Keypoints
from torch import nn
from utility.functions import cos_similarity, gauss


class PassingDetector:
    def __init__(self, config_path: str, **defs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.model = LSTMModel(**cfg)
        self.defaults = defs

        param = torch.load(cfg["pretrained_path"])
        self.model.load_state_dict(param)
        self.model.to(self.device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def extract_feature(self, p1, p2, que):
        mu = self.defaults["gauss_mu"]
        sigma = self.defaults["gauss_sig"]
        wrist_mu = self.defaults["wrist_gauss_mu"]
        wrist_sig = self.defaults["wrist_gauss_sig"]
        seq_len = self.defaults["seq_len"]

        # get indicator
        def get_indicators(person):
            kps = Keypoints(person["keypoints"])
            pos = person["position"]
            body = person["body"]
            arm = person["arm"]
            wrist = (kps.get("lwrist", True), kps.get("rwrist", True))
            ret = SimpleNamespace(
                **{"pos": pos, "body": body, "arm": arm, "wrist": wrist}
            )
            return ret

        p1_data = get_indicators(p1)
        p2_data = get_indicators(p2)

        if None in p1_data.__dict__.values() or None in p2_data.__dict__.values():
            return None

        # calc distance of position
        p1_pos = np.array(p1_data.pos)
        p2_pos = np.array(p2_data.pos)

        norm = np.linalg.norm(p1_pos - p2_pos, ord=2)
        distance = gauss(norm, mu=mu, sigma=sigma)

        p1p2 = p2_pos - p1_pos
        p2p1 = p1_pos - p2_pos

        p1p2_sim = cos_similarity(p1_data.body, p1p2)
        p2p1_sim = cos_similarity(p2_data.body, p2p1)
        body_distance = (np.average([p1p2_sim, p2p1_sim]) + 1) / 2

        # calc arm average
        arm_ave = np.average([p1_data.arm, p2_data.arm])

        # calc wrist distance
        min_norm = np.inf
        for i in range(2):
            for j in range(2):
                norm = np.linalg.norm(
                    np.array(p1_data.wrist[i]) - np.array(p2_data.wrist[j]), ord=2
                )
                if norm < min_norm:
                    min_norm = norm

        wrist_distance = gauss(min_norm, mu=wrist_mu, sigma=wrist_sig)

        # concatnate to feature
        feature = [distance, body_distance, arm_ave, wrist_distance]
        que.append(feature)

        return que[-seq_len:]

    def predict(self, features):
        with torch.no_grad():
            features = torch.Tensor(np.array([features])).float().to(self.device)
            pred = self.model(features)
            pred = pred.max(1)[1]
            pred = pred.cpu().numpy()[0]

        return pred


class LSTMModel(nn.Module):
    def __init__(self, **config):
        super(LSTMModel, self).__init__()

        # init rnn layers
        in_dim = config["size"]
        out_dim = config["rnn_hidden_dim"]
        n_rnns = config["n_rnns"]
        rnn_dropout = config["rnn_dropout"]
        self.rnn = nn.LSTM(
            in_dim, out_dim, num_layers=n_rnns, dropout=rnn_dropout, batch_first=True
        )

        # init linear layers
        self.linears = nn.Sequential()
        for i in range(config["n_linears"]):
            if i == 0:
                in_dim = config["rnn_hidden_dim"]
            else:
                in_dim = config["hidden_dims"][i - 1]

            out_dim = config["hidden_dims"][i]
            dropout = config["dropouts"][i]
            self.linears.add_module(f"fc{i + 1}", Linear(in_dim, out_dim, dropout))

        # init output layers
        self.output_layer = nn.Linear(config["hidden_dims"][-1], config["n_classes"])
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x, (_, _) = self.rnn(x)
        x = x[:, -1, :]
        x = self.linears(x)
        x = self.output_layer(x)
        x = self.softmax(x)

        return x


class Linear(nn.Sequential):
    def __init__(self, in_dim, out_dim, dropout):
        super(Linear, self).__init__(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
