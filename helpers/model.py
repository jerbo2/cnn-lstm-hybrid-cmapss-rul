import torch.nn as nn

class RULPredictionModel(nn.Module):
    def __init__(self, num_features, hidden_size, num_lstms, num_classes=1):
        super(RULPredictionModel, self).__init__()
        self.num_lstms = num_lstms

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=num_features, out_channels=64, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=5, padding=2
        )
        self.conv4 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=5, padding=2
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        for i in range(num_lstms):
            if i == 0:
                setattr(
                    self,
                    f"lstm{i+1}",
                    nn.LSTM(input_size=256, hidden_size=hidden_size, batch_first=True),
                )
            else:
                setattr(
                    self,
                    f"lstm{i+1}",
                    nn.LSTM(
                        input_size=hidden_size,
                        hidden_size=hidden_size,
                        batch_first=True,
                    ),
                )

        # Dense layer
        self.fc = nn.Linear(hidden_size, num_classes)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape input data (batch_size, num_features, sequence_length)
        x = x.transpose(1, 2)

        # Convolutional layers with ReLU activations
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool2(x)

        # Dropout layer
        x = self.dropout(x)

        # Reshape output for LSTM layers
        # We need to make sure the LSTM input is (batch_size, seq_len, num_features)
        x = x.transpose(1, 2)

        # LSTM layers
        for i in range(self.num_lstms):
            x, _ = getattr(self, f"lstm{i+1}")(x)

        # Dropout layer
        x = self.dropout(x)

        # Take the output of the last time step
        x = x[:, -1, :]

        # Fully connected layer
        x = self.fc(x)
        return x

