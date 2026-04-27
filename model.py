import torch
import torch.nn as nn
def double_convolution(in_channels: int, out_channels: int):
  #this represents a single "block" of the U-Net architecture. padding is added
  #so that the output of the network is still 572x572
  conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(out_channels),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(out_channels)
  )
  return conv

class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)
        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2,
            stride=2,
            output_padding=1) # Added output_padding so that the dimensions return properly
        # Below, `in_channels` again becomes 1024 as we are concatenating.
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2,
            output_padding=1) # Added output_padding
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_convolution_4 = double_convolution(128, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )
    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        #print(down_1.shape)
        down_2 = self.max_pool2d(down_1)
        #print(down_2.shape)
        down_3 = self.down_convolution_2(down_2)
        #print(down_3.shape)
        down_4 = self.max_pool2d(down_3)
        #print(down_4.shape)
        down_5 = self.down_convolution_3(down_4)
        #print(down_5.shape)
        down_6 = self.max_pool2d(down_5)
        #print(down_6.shape)
        down_7 = self.down_convolution_4(down_6)
        #print(down_7.shape)
        down_8 = self.max_pool2d(down_7)
        #print(down_8.shape)
        down_9 = self.down_convolution_5(down_8)
        #print(down_9.shape)
        # *** DO NOT APPLY MAX POOL TO down_9 ***

        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        out = self.out(x)
        return out


if __name__ == '__main__':
    print("This is just a model definition file! It can't be run on it's own :( ")

