import torch
import torch.nn as nn

class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()

        # Encoding
        self.enc0 = self._preact_block(1, 16)
        self.enc1 = self._preact_block(16, 32)
        self.enc2 = self._preact_block(32, 64)
        self.enc3 = self._preact_block(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Decoding (upsample + conv after skip)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = self._preact_block(128, 64)  # 64 upsampled + 64 skip

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = self._preact_block(64, 32)   # 32 upsampled + 32 skip

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv1 = self._preact_block(32, 16)    # 16 upsampled + 16 skip

        # Final output layer
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    #Preactivation block that nomralises and activates before convolution
    def _preact_block(self, in_ch, out_ch, dropout_p=0.2):
        return nn.Sequential(
            nn.InstanceNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p)
        )

    def forward(self, x):
        # Encoder
        x0 = self.enc0(x)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # Decoder
        d3 = self.up3(x3)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.conv2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.conv1(d1)

        out = self.final(d1)
        out = torch.sigmoid(out)
        return out


#DiceLoss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return 1 - dice
