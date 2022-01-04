import numpy as np
import torch
import yaml
from common.default import PASSING_DEFAULT
from common.functions import cos_similarity, gauss
from common.json import IA_FORMAT, START_IDX
from common.keypoint import body
from torch import nn


class PassingDetector:
    def __init__(self, config_path, checkpoint_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.model = RNNModel(**cfg)
        self.seq_len = cfg["seq_len"]

        param = torch.load(checkpoint_path)
        self.model.load_state_dict(param)
        self.model.to(self.device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def extract_feature(
        self,
        p1,
        p2,
        queue,
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
        p1_wrist = (p1_kps[body["LWrist"]][:2], p1_kps[body["RWrist"]][:2])
        p2_wrist = (p2_kps[body["LWrist"]][:2], p2_kps[body["RWrist"]][:2])

        if (
            p1_pos is None
            or p2_pos is None
            or p1_body is None
            or p2_body is None
            or p1_arm is None
            or p2_arm is None
            or p1_wrist is None
            or p2_wrist is None
        ):
            return None

        p1_pos = np.array(p1_pos)
        p2_pos = np.array(p2_pos)

        # calc distance
        norm = np.linalg.norm(p1_pos - p2_pos, ord=2)
        distance = gauss(norm, mu=mu, sigma=sigma)

        # calc vector of each other
        p1p2 = p2_pos - p1_pos
        p2p1 = p1_pos - p2_pos

        p1p2_sim = cos_similarity(p1_body, p1p2)
        p2p1_sim = cos_similarity(p2_body, p2p1)
        body_direction = (np.average([p1p2_sim, p2p1_sim]) + 1) / 2

        # calc arm average
        arm_ave = np.average([p1_arm, p2_arm])

        # calc wrist distance
        min_norm = np.inf
        for i in range(2):
            for j in range(2):
                norm = np.linalg.norm(
                    np.array(p1_wrist[i]) - np.array(p2_wrist[j]), ord=2
                )
                if norm < min_norm:
                    min_norm = norm

        wrist_distance = gauss(min_norm, mu=wrist_mu, sigma=wrist_sig)

        feature = [distance, body_direction, arm_ave, wrist_distance]

        queue.append(feature)

        return queue[-self.seq_len :]

    def predict(self, features, pass_duration, length=5):
        if len(features) < self.seq_len:
            return 0, pass_duration

        with torch.no_grad():
            features = torch.Tensor(np.array([features])).float().to(self.device)
            pred = self.model(features)
            pred = pred.max(1)[1]
            pred = pred.cpu().numpy()[0]

        if pred == 1:
            pass_duration += 1

        else:
            pass_duration = 0

        # return (
        #     1 if pass_duration > length else 0,
        #     pass_duration
        # )
        return pred, pass_duration


class RNNModel(nn.Module):
    def __init__(self, **config):
        super(RNNModel, self).__init__()

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
