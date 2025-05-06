import torch
import torch.nn as nn

# -----------------------------------
# Object Detection Head (Anchor Free)
# -----------------------------------
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        # sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class DetectionHead(nn.Module):
    """
    • This detection head module contains 3 heads, each takes input from
      different levels/ scales of light-weight refinenet network. Specifically [l3, l5, l7].
    • Each of the 3 heads are decoupled (i.e. seperate bounding box block and classification block)
    • This module is an implementation of YoloV8 head. Made changes to integrate it with the Multi-Task Network.
    
    Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py

    """
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(
        self,
        num_classes=80,
        decoder_channels=(256, 256, 256),
        head_channels=(64, 128, 256),
        stride=(8, 16, 32),
        reg_max=16
    ):
        """
        Parameters:
        ----------
        
        num_classes: int
            The number of classes
        decoder_channels: tuple
            A tuple of decoder (LIGHT-WEIGHT REFINENET) out-channels at various layers l3, l5, l7 respectively
        head_channels: tuple
            A tuple of input-channels for the 3 heads of the detecion head
        stride: tuple
            A tuple of strides at different scales
        reg_max: int
            Anchor scale factor
        """
        
        super().__init__()

        self.num_classes = num_classes  # number of classes
        self.decoder_channels = decoder_channels
        self.reg_max = reg_max # DFL channels (head_channels[0] // 16 to scale: 4)
        
        self.num_heads = len(head_channels)  # num of detection heads
        self.no = self.num_classes + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor(stride)  # strides will be computed during build using this (only once)
        
        self.p = nn.ModuleList(
            nn.Conv2d(in_, out_, 3, padding=1)
            for in_, out_ in zip(decoder_channels, head_channels)
        ) # [P3, P5, P7]
        
        self.pool = nn.MaxPool2d(2)
        
        c2 = max((16, head_channels[0] // 4, self.reg_max * 4))
        c3 = max(head_channels[0], self.num_classes)

        self.bbox_layers = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) 
            for x in head_channels
        )   # [P3, P5, P7]
        
        self.class_layers = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.num_classes, 1)) 
            for x in head_channels
        )   # [P3, P5, P7]
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):

        for i in range(self.num_heads):
            xin = self.pool(self.p[i](x[i]))
            box_out = self.bbox_layers[i](xin)  # box_out_channels = 4 * self.reg_max = 64
            cls_out = self.class_layers[i](xin) # cls_out_channels = no of classes 
            x[i] = torch.cat((box_out, cls_out), 1) # N_CHANNELS = self.no = box_out_channels + cls_out_channels

        if self.training:
            # For input imahe height = 192, width = 640
            # x[0] shape = (batch_size, self.no, 24, 80)
            # x[1] shape = (batch_size, self.no, 12, 40)
            # x[2] shape = (batch_size, self.no, 6, 20)
            return x

        # N_OUT = (6 * 20) + (12 * 40) + (24 * 80) = 2520

        shape = x[0].shape  # BCHW
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
            # shape: self.anchors = (2, N_OUT); self.strides = (1, N_OUT)

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) # shape: (batch_size, self.no, N_OUT)

        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1) 
        # shape: box = (batch_size, 4 * self.reg_max, N_OUT); cls = (batch_size, no. of classes, N_OUT)

        dbox = dist2bbox(
            self.dfl(box),
            self.anchors.unsqueeze(0),
            xywh=True,
            dim=1
        ) * self.strides

        # shape: dbox = (batch_size, 4, N_OUT)

        y = torch.cat((dbox, cls.sigmoid()), 1)  # shape: (batch_size, 4 + no. of classes, N_OUT). e.g (2, 18, 2520)

        return y, x
