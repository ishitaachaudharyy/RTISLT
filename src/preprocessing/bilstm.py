import torch
import torch.nn as nn


class ISLBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=225,
        hidden_size=128,
        num_layers=2,
        num_classes=50,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        final_hidden = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        logits = self.classifier(
            final_hidden
        )

        return logits