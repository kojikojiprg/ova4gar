from torch import nn


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.seq_len = kwargs["seq_len"]
        self.n_features = kwargs["n_features"]
        self.z_size = kwargs["z_size"]

        self.rnn1 = nn.LSTM(
            input_size=kwargs["n_features"],
            hidden_size=kwargs["hidden_size1"],
            num_layers=kwargs["n_layers"],
            dropout=kwargs["dropout"],
            batch_first=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=kwargs["hidden_size1"],
            hidden_size=kwargs["hidden_size2"],
            num_layers=kwargs["n_layers"],
            dropout=kwargs["dropout"],
            batch_first=True,
        )
        self.rnn3 = nn.LSTM(
            input_size=kwargs["hidden_size2"],
            hidden_size=kwargs["z_size"],
            num_layers=kwargs["n_layers"],
            dropout=kwargs["dropout"],
            batch_first=True,
        )

    def forward(self, x):
        x = x.reshape((-1, self.seq_len, self.n_features))
        # x.shape = (batch_size, seq_len, n_features)

        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        x, (z, _) = self.rnn3(x)
        # x.shape = (batch_size, seq_len, z_size)
        # z.shape = (1, batch_size, z_size)

        return z.reshape(-1, self.z_size)


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.seq_len = kwargs["seq_len"]
        self.z_size = kwargs["z_size"]
        self.n_layers = kwargs["n_layers"]

        self.rnn1 = nn.LSTM(
            input_size=kwargs["z_size"],
            hidden_size=kwargs["z_size"],
            num_layers=kwargs["n_layers"],
            dropout=kwargs["dropout"],
            batch_first=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=kwargs["z_size"],
            hidden_size=kwargs["hidden_size2"],
            num_layers=kwargs["n_layers"],
            dropout=kwargs["dropout"],
            batch_first=True,
        )
        self.rnn3 = nn.LSTM(
            input_size=kwargs["hidden_size2"],
            hidden_size=kwargs["hidden_size1"],
            num_layers=kwargs["n_layers"],
            dropout=kwargs["dropout"],
            batch_first=True,
        )
        self.output_layer = nn.Linear(kwargs["hidden_size1"], kwargs["n_features"])

    def forward(self, z):
        z = z.repeat(1, self.seq_len).reshape(-1, self.seq_len, self.z_size)
        # z.shape = (batch_size, seq_len, self.z_size)

        x, (_, _) = self.rnn1(z)
        x, (_, _) = self.rnn2(x)
        x, (_, _) = self.rnn3(x)
        x = self.output_layer(x)
        # x.shape = (batch_size, seq_len, n_feature)

        return x


class RecurrentAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(**kwargs).to(kwargs["device"])
        self.decoder = Decoder(**kwargs).to(kwargs["device"])

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z
