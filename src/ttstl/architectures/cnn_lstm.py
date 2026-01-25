import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM neural network architecture for time series forecasting.

    This model combines 1D convolutions for feature extraction from multiple
    input channels with LSTM-based temporal modeling. It includes Inception-like
    modules to capture features at different scales.

    Architecture:
    1. Initial convolution block (Conv2d -> LeakyReLU -> BatchNorm)
    2. Deep convolution block with varying kernel sizes
    3. Three parallel Inception branches extracting features with different kernel sizes
    4. Concatenation of Inception outputs
    5. Permutation and dimensions adjustment
    6. LSTM layer processing the feature sequence
    7. Linear output layer mapping to scalar prediction

    Parameters
    ----------
    None
        Parameters are hardcoded in the current implementation.

    Attributes
    ----------
    conv1 : nn.Sequential
        First convolutional block.
    conv2 : nn.Sequential
        Second convolutional block.
    inp1 : nn.Sequential
        First Inception branch (kernel size 1 and 3).
    inp2 : nn.Sequential
        Second Inception branch (kernel size 1 and 5).
    inp3 : nn.Sequential
        Third Inception branch (MaxPooling and kernel 1).
    lstm : nn.LSTM
        Long Short-Term Memory layer.
    output_layer : nn.Linear
        Final fully connected layer.
    """

    def __init__(self):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
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
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
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
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(
            input_size=192, hidden_size=64, num_layers=1, batch_first=True
        )

        self.output_layer = nn.Linear(64, 1)

    def _extract_features(self, x, h0=torch.zeros(1, 64), c0=torch.zeros(1, 64)):
        """
        Extract features from input tensor using CNN and LSTM layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, height, width) or (batch, 1, height, width).
        h0 : torch.Tensor, optional
            Initial hidden state for LSTM. Default is zeros.
        c0 : torch.Tensor, optional
            Initial cell state for LSTM. Default is zeros.

        Returns
        -------
        torch.Tensor
            Feature vector extracted from the last LSTM step.
        """

        # Add a channel dimension
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
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model prediction (scalar).
        """
        x = self._extract_features(x)
        x = self.output_layer(x)
        return x

    def hidden_activations(self, x):
        """
        Extract hidden activations (features) without the final linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Feature embeddings from the LSTM layer.
        """
        x = self._extract_features(x)
        return x
