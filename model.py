import torch
import torch.nn as nn
def double_convolution(in_channels: int, out_channels: int):
  #this represents a single "block" of the U-Net architecture. padding is added
  #so that the output of the network is still 576x576
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
    def __init__(self, num_classes=1, verbose=False):
        super(UNet, self).__init__()
        self.verbose = verbose
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
            stride=2,) # Removed output padding because There's no real need to stick to 572x572 when it just makes things kind of weird.
        # Below, `in_channels` again becomes 1024 as we are concatenating.
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2,)
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
        if self.verbose: print(f"Input: {x.shape}")
        down_1 = self.down_convolution_1(x)
        if self.verbose: print("Encoder Path:")
        if self.verbose: print(f"Double Convolution 1: {down_1.shape}")
        down_2 = self.max_pool2d(down_1)
        if self.verbose: print(f"Max-Pool 1: {down_2.shape}")
        down_3 = self.down_convolution_2(down_2)
        if self.verbose: print(f"Double Convolution 2: {down_3.shape}")
        down_4 = self.max_pool2d(down_3)
        if self.verbose: print(f"Max-Pool 2: {down_4.shape}")
        down_5 = self.down_convolution_3(down_4)
        if self.verbose: print(f"Double Convolution 3: {down_5.shape}")
        down_6 = self.max_pool2d(down_5)
        if self.verbose: print(f"Max-Pool 3: {down_6.shape}")
        down_7 = self.down_convolution_4(down_6)
        if self.verbose: print(f"Double Convolution 4: {down_7.shape}")
        down_8 = self.max_pool2d(down_7)
        if self.verbose: print(f"Max-Pool 4: {down_8.shape}")
        down_9 = self.down_convolution_5(down_8)
        if self.verbose: print(f"Double Convolution 5: {down_9.shape}")

        if self.verbose: print("\nDecoder Pass:")
        up_1 = self.up_transpose_1(down_9)
        if self.verbose: print(f"Up-Conv 1: {up_1.shape}")
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        if self.verbose: print(f"Cat + Double Conv 1: {x.shape}")
        up_2 = self.up_transpose_2(x)
        if self.verbose: print(f"Up-Conv 2: {up_2.shape}")
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        if self.verbose: print(f"Cat + Double Conv 2: {x.shape}")
        up_3 = self.up_transpose_3(x)
        if self.verbose: print(f"Up-Conv 3: {up_3.shape}")
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        if self.verbose: print(f"Cat + Double Conv 3: {x.shape}")
        up_4 = self.up_transpose_4(x)
        if self.verbose: print(f"Up-Conv 4: {up_4.shape}")
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        if self.verbose: print(f"Cat + Double Conv 4: {x.shape}")
        out = self.out(x)
        if self.verbose: print(f"Output: {out.shape}")
        return out

    def dummy_pass(self):
        tensor = torch.rand(1,3,512,512)
        self.forward(tensor)


if __name__ == '__main__':
    print("This is just a model definition file! It doesn't do much on it's own :( ")
    print("Have this readout of the model's dimensions though! :D")
    model = UNet(verbose=True)
    model.dummy_pass()
