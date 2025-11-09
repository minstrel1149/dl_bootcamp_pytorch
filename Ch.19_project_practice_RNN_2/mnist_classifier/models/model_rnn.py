import torch
import torch.nn as nn

class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout_p):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=n_layers,
            # batch_first는 무조건 True로 가져간다고 생각해야
            # 양방향 LSTM이므로 bidirectional True로
            batch_first=True, bidirectional=True,
            dropout=dropout_p
        )

        self.layers = nn.Sequential(
            nn.LeakyReLU(),
            # 양방향 LSTM이므로 hidden_size에 2를 곱해줘야
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, X):
        # LSTM은 hidden state, cell state가 함께 튜플로 반환
        z, _ = self.lstm(X)
        # many to one 형태이므로 마지막 순서의 값만 얻어오는 형태
        z = z[:, -1]
        y = self.layers(z)

        return y