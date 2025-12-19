import torch
import torch.nn as nn

class LargeKernelBlock(nn.Module):
    """
    Large Kernel Block for Depth Stream
    Paper Reference: HDBFormer
    Structure: DW-Conv(7x7) -> BN -> ReLU -> PW-Conv(1x1) -> BN
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super().__init__()
        
        # Depthwise Conv (Large Kernel)
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        
        # Pointwise Conv
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection if in == out and stride == 1
        self.use_res = (in_channels == out_channels) and (stride == 1)

    def forward(self, x):
        residual = x
        
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.pw_conv(out)
        out = self.bn2(out)
        
        if self.use_res:
            out += residual
            
        return out
