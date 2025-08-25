import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, 
                kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 10)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Inception modules.
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, 
                kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, 
                kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, 
                kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, 
                kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=32, out_channels=64, 
                kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(
            input_size=192, hidden_size=64, num_layers=1, batch_first=True
        )

        self.output_layer = nn.Linear(64, 1)
    
    def _extract_features(self, x, 
                h0 = torch.zeros(1, 64), 
                c0 = torch.zeros(1, 64)):

        #Add a channel dimension
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(3)

        self.lstm.flatten_parameters()

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        return x
    
    def forward(self, x):
        x = self._extract_features(x)
        x = self.output_layer(x)
        return x

    def hidden_activations(self, x):
        x = self._extract_features(x)
        return x