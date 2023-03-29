import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, dropout_rate=0.2):

        super(CNNLSTM, self).__init__()

        self.dropout_rate = dropout_rate

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Flatten()
        )
        self.fc_conv = nn.Linear(in_features=144, out_features=32)

        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc1 = nn.Linear(in_features=64, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=2)
        self.relu = nn.ReLU()

    def forward(self, x1, x2) -> torch.Tensor:
        conv_out = self.fc_conv(self.dropout(self.relu(self.conv(x1))))

        output, (h_n, c_n) = self.lstm(x2)
        lstm_out = self.relu(h_n[-1, :, :])

        out = torch.cat([conv_out, lstm_out], dim=1)
        out = self.fc2(self.dropout(self.relu(self.fc1(out))))

        return out
