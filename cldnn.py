import torch.nn as nn
import torch
class CLDNN(nn.Module):
    def __init__(self, num_cls=8):
        super(CLDNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 2 * 25, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(2 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 4)),

            nn.Conv2d(2 * 25, 4 * 25, kernel_size=(1, 11), padding=(0, 5), bias=False),
            nn.BatchNorm2d(4 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 4)),

            nn.Conv2d(4 * 25, 8 * 25, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(8 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(8 * 25, 10 * 25, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(10 * 25),
            nn.ReLU(inplace=True),
            nn.Conv2d(10 * 25, 10 * 25, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(10 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(10 * 25, 12 * 25, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(12 * 25),
            nn.ReLU(inplace=True),
            nn.Conv2d(12 * 25, 12 * 25, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(12 * 25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),
        )
        self.My_LSTM = nn.LSTM(12 * 25, 3 * 25, num_layers=1, batch_first=True,
                               bidirectional=True)  # , dropout=0.2, dropout=0.5
        # 如果是True，则input为(batch, seq, input_size)。默认值为：False（seq_len, batch, input_size）
        self.classifier1 = nn.Linear(3 * 25 * 4, num_cls)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        # print(x.shape) # torch.Size([256, 300, 1, 8])
        x = torch.squeeze(x)
        x = x.transpose(1, 2)
        x, _ = self.My_LSTM(x, None)
        fature = torch.cat([x[:, 0, :], x[:, -1, :]], 1)
        x = self.classifier1(fature)
        return x, fature
