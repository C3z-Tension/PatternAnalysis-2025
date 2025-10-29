import torch
import torch.nn as nn

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
    
class ImprovedUNet(nn.Module):
    def __init__(self):
        super(ImprovedUNet, self).__init__()
        
        #Encoding
        self.enc1 = self._preact_block(1, 64)
        self.enc2 = self._preact_block(64, 128)
        self.enc3 = self._preact_block(128, 256)

        #Decoding
        #self.dec1 = 

        #Pooling
        #Upsamplig
        #Sigmoid (?)

    def _preact_block(self, in_ch, out_ch, dropout_p=0.2):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        # Continue with pooling, upsampling, skip connections, etc.
        return x3



#DiceLoss function