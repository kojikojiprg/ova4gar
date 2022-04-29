from torch import nn


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
