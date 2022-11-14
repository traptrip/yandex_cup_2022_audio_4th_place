from torch import nn
from torch.nn import functional as F


class BasicNet(nn.Module):
    def __init__(self, input_features_size, output_features_size):
        super().__init__()
        self.input_features_size = input_features_size
        self.output_features_size = output_features_size
        self.conv_0 = nn.Conv1d(input_features_size, 1024, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(
            512, 512, kernel_size=3, padding=1
        )
        self.mp_1 = nn.MaxPool1d(2, 2)
        self.conv_3 = nn.Conv1d(
            512, output_features_size, kernel_size=3, padding=1
        )
        self.conv_4 = nn.Conv1d(
            output_features_size, output_features_size, kernel_size=3, padding=1
        )
        self.activation = nn.LeakyReLU()


    def forward(self, x):
        identity = x

        x = self.conv_0(x)
        x = self.activation(x)
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x += identity
        x = self.activation(x)

        x = self.mp_1(x)

        x = self.conv_3(x)
        x = self.activation(x)
        x = self.conv_4(x).mean(axis=2)

        return x


class BasicNetV2(nn.Module):
    def __init__(self, output_features_size=512):
        super().__init__()
        self.output_features_size = output_features_size
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                512, 8, dim_feedforward=2048, dropout=0.2, batch_first=True
            ),
            num_layers=3
        )
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        x = x.permute((0, 2, 1))
        x = self.transformer(x.float())
        x = x.permute((0, 2, 1))
        x = self.avg_pooling(x).squeeze()
        return x


class BasicNetV3(nn.Module):
    def __init__(self, output_features_size=512):
        super().__init__()
        self.output_features_size = output_features_size
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                output_features_size, 8, dim_feedforward=2048, dropout=0.25, batch_first=True
            ),
            num_layers=1
        )
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        x = x.permute((0, 2, 1))
        x = self.transformer(x.float())
        x = x.permute((0, 2, 1))
        x = self.avg_pooling(x).squeeze()
        return x
