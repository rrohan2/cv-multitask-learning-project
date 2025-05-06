# ssd_hydranet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------
# Basic Convolution Blocks
# --------------------------

def conv3x3(in_channels, out_channels, stride=1, dilation=1, groups=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=bias, groups=groups)

def batchnorm(num_features):
    return nn.BatchNorm2d(num_features, affine=True, eps=1e-5, momentum=0.1)

def convbnrelu(in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=int(kernel_size / 2.), groups=groups, bias=False),
        batchnorm(out_channels)
    ]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


# Continuation of ssd_hydranet_model.py

# --------------------------
# HydraNet Encoder
# --------------------------

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super().__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(
            convbnrelu(in_planes, intermed_planes, 1),
            convbnrelu(intermed_planes, intermed_planes, 3, stride=stride, groups=intermed_planes),
            convbnrelu(intermed_planes, out_planes, 1, act=False)
        )

    def forward(self, x):
        out = self.output(x)
        return out + x if self.residual else out

class HydraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tasks = 2
        self.num_classes = 6
        self.define_mobilenet()
        self.define_lightweight_refinenet()

    def define_mobilenet(self):
        config = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
                  [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        self.in_channels = 32
        self.layer1 = convbnrelu(3, self.in_channels, 3, 2)
        c_layer = 2
        for t, c, n, s in config:
            layers = []
            for i in range(n):
                layers.append(InvertedResidualBlock(self.in_channels, c, t, s if i == 0 else 1))
                self.in_channels = c
            setattr(self, f'layer{c_layer}', nn.Sequential(*layers))
            c_layer += 1

    def _make_crp(self, in_planes, out_planes, stages, groups=False):
        return nn.Sequential(CRPBlock(in_planes, out_planes, stages, groups=groups))

    def define_lightweight_refinenet(self):
        self.conv8 = conv1x1(320, 256)
        self.conv7 = conv1x1(160, 256)
        self.conv6 = conv1x1(96, 256)
        self.conv5 = conv1x1(64, 256)
        self.conv4 = conv1x1(32, 256)
        self.conv3 = conv1x1(24, 256)
        self.crp4 = self._make_crp(256, 256, 4)
        self.crp3 = self._make_crp(256, 256, 4)
        self.crp2 = self._make_crp(256, 256, 4)
        self.crp1 = self._make_crp(256, 256, 4, groups=True)
        self.conv_adapt4 = conv1x1(256, 256)
        self.conv_adapt3 = conv1x1(256, 256)
        self.conv_adapt2 = conv1x1(256, 256)
        self.pre_depth = conv1x1(256, 256, groups=256)
        self.depth = conv3x3(256, 1, bias=True)
        self.pre_segm = conv1x1(256, 256, groups=256)
        self.segm = conv3x3(256, self.num_classes, bias=True)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x); l3 = self.layer3(x)
        l4 = self.layer4(l3); l5 = self.layer5(l4)
        l6 = self.layer6(l5); l7 = self.layer7(l6); l8 = self.layer8(l7)

        ol3, ol4, ol5, ol6, ol7, ol8 = l3, l4, l5, l6, None, l8

        l8 = self.conv8(l8); l7 = self.conv7(l7); l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7); l7 = self.conv_adapt4(l7)
        l7 = F.interpolate(l7, size=l6.size()[2:], mode='bilinear', align_corners=False)
        ol7 = l7

        l6 = self.conv6(l6); l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7); l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = F.interpolate(l5, size=l4.size()[2:], mode='bilinear', align_corners=False)

        l4 = self.conv4(l4); l4 = self.relu(l5 + l4); l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = F.interpolate(l4, size=l3.size()[2:], mode='bilinear', align_corners=False)

        l3 = self.conv3(l3); l3 = self.relu(l3 + l4); l3 = self.crp1(l3)

        out_segm = self.segm(self.relu(self.pre_segm(l3)))
        out_d = self.depth(self.relu(self.pre_depth(l3)))

        if hasattr(self, "object_detection_head") and self.object_detection_head == 1:
            return out_segm, out_d, (ol3, ol4, ol5, ol6, ol7, ol8)
        return out_segm, out_d

class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super().__init__()
        for i in range(n_stages):
            setattr(self, f'{i + 1}_outvar_dimred',
                    conv1x1(in_planes if i == 0 else out_planes, out_planes,
                            groups=in_planes if groups else 1))
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(5, 1, 2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, f'{i + 1}_outvar_dimred')(top)
            x = top + x
        return x


# --------------------------
# SSD Helper Functions
# --------------------------

def cxcy_to_xy(bboxes):
    return torch.cat([bboxes[:, :2] - (bboxes[:, 2:] / 2),
                      bboxes[:, :2] + (bboxes[:, 2:] / 2)], 1)

def decode_bboxes(offsets, default_boxes):
    return torch.cat([offsets[:, :2] * default_boxes[:, 2:] / 10 + default_boxes[:, :2],
                      torch.exp(offsets[:, 2:] / 5) * default_boxes[:, 2:]], 1)

def intersect(boxes1, boxes2):
    n1, n2 = boxes1.size(0), boxes2.size(0)
    max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2))
    min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
                       boxes2[:, :2].unsqueeze(0).expand(n1, n2, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def find_IoU(boxes1, boxes2):
    inter = intersect(boxes1, boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    area1 = area1.unsqueeze(1).expand_as(inter)
    area2 = area2.unsqueeze(0).expand_as(inter)
    union = area1 + area2 - inter
    return inter / union

class L2Norm(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1, channels, 1, 1))
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt()
        return x / norm * self.weight

# --------------------------
# SSD300MB9 Class
# --------------------------

class SSD300MB9(nn.Module):
    """ Add Custom SSD head on top of ol4 and ol7 from existing hydranet. 9th Iteration. """

    def __init__(self, num_classes, hydra):
        super(SSD300MB9, self).__init__()
        self.num_classes = num_classes
        self.base_model = hydra              #The earlier HydraNet will be our "base model"

        self.L2Norm = L2Norm(channels= 512, scale= 20)

        # The boxes to generate for the different resolutions/zoom
        num_boxes = {"conv4_3": 4, "conv7": 6, "conv8_2": 6,
                     "conv9_2": 6, "conv10_2": 4, "conv11_2": 4}

        # Auxiliary Network
        self.conv4_3_out = nn.Conv2d(32, 512, kernel_size= 1, padding=0, stride= 1)

        self.conv7_out = nn.Conv2d(256, 1024, kernel_size= 1, padding=0, stride= 1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=(1, 1), padding= 0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size= (3, 3), padding= 1, stride= 2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size= (1, 1), padding= 0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 1, stride= 2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size= (1, 1), padding= 0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size= (1, 1), padding= 0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size= (3, 3), padding= 0)

        # Prediction Network
        self.conv_4_3_loc = nn.Conv2d(512, num_boxes["conv4_3"] * 4,
                                      kernel_size= (3, 3), padding= 1)
        self.conv7_loc = nn.Conv2d(1024, num_boxes["conv7"] * 4,
                                   kernel_size= (3, 3), padding= 1)
        self.conv8_2_loc = nn.Conv2d(512, num_boxes["conv8_2"]*4,
                                     kernel_size= (3, 3), padding= 1)
        self.conv9_2_loc = nn.Conv2d(256, num_boxes["conv9_2"]*4,
                                     kernel_size= (3, 3), padding= 1)
        self.conv10_2_loc = nn.Conv2d(256, num_boxes["conv10_2"]*4,
                                      kernel_size= (3, 3), padding= 1)
        self.conv11_2_loc = nn.Conv2d(256, num_boxes["conv11_2"]*4,
                                      kernel_size= (3, 3), padding= 1)

        #Class Prediction Network
        self.conv4_3_cls = nn.Conv2d(512, num_boxes["conv4_3"]*num_classes,
                                     kernel_size= (3, 3), padding= 1)
        self.conv7_cls = nn.Conv2d(1024, num_boxes["conv7"] * num_classes,
                                   kernel_size= (3, 3), padding= 1)
        self.conv8_2_cls = nn.Conv2d(512, num_boxes["conv8_2"]*num_classes,
                                     kernel_size= (3, 3), padding= 1)
        self.conv9_2_cls = nn.Conv2d(256, num_boxes["conv9_2"]*num_classes,
                                     kernel_size= (3, 3), padding= 1)
        self.conv10_2_cls = nn.Conv2d(256, num_boxes["conv10_2"]*num_classes,
                                      kernel_size= (3, 3), padding= 1)
        self.conv11_2_cls = nn.Conv2d(256, num_boxes["conv11_2"]*num_classes,
                                      kernel_size= (3, 3), padding= 1)

        #num_default_boxes = 8732
        self.default_boxes = self.create_default_boxes()

    def forward(self, image):
        """ Forward propagation
            image: images, a tensor of dimension (N, 3, 300, 300)

            Out: hydra_segm, hydra_depth, 8732 pred loc, classes for each image
        """

        #Set debugmode to 1 to see the size of tensors at different layers
        debugmode = 0

        #Extract Layers from MobileNetV2 in the pretrained HydraNet
        segm, depth, object_detection_inputlayers = self.base_model(image)
        (l3,l4,l5,l6,l7,l8) = object_detection_inputlayers

        batch_size = l8.size(0)

        if debugmode==1:
            print("l3",l3.size())
            print("l4",l4.size())
            print("l5",l5.size())
            print("l6",l6.size())
            print("l7",l7.size())
            print("l8",l8.size())

        conv4_3_out = F.relu(self.conv4_3_out(l4))
        conv4_3_out = self.L2Norm(conv4_3_out)
        if debugmode==1:
            print("conv4_3_out",conv4_3_out.size())

        conv7_out = F.relu(self.conv7_out(l7))
        if debugmode==1:
            print("CONV7_out",conv7_out.size())

        # AUXILIARY NETWORK
        x = conv7_out    #(N, 1024, 19, 19)...
        x = F.relu(self.conv8_1(x))    #(N, 256, 19, 19)
        x = F.relu(self.conv8_2(x))    #(N, 512, 10, 10)
        conv8_2_out = x
        if debugmode==1:
            print("conv8_2_out",conv8_2_out.size())

        x = F.relu(self.conv9_1(x))    #(N, 128, 10, 10)
        x = F.relu(self.conv9_2(x))    #(N, 256, 5, 5)
        conv9_2_out = x
        if debugmode==1:
            print("conv9_2_out",conv9_2_out.size())

        x = F.relu(self.conv10_1(x))   #(N, 128, 5, 5)
        x = F.relu(self.conv10_2(x))   #(N, 256, 3, 3)
        conv10_2_out = x
        if debugmode==1:
            print("conv10_2_out",conv10_2_out.size())

        x = F.relu(self.conv11_1(x))   #(N, 128, 3, 3)
        conv11_2_out = F.relu(self.conv11_2(x))   #(N, 256, 1, 1)
        if debugmode==1:
            print("conv11_2_out",conv11_2_out.size())

        # PREDICTION NETWORK -- Predictions for the groups of "Anchor Box" at different scale
        l_conv4_3 = self.conv_4_3_loc(conv4_3_out)    #(N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()    #(N, 38, 38, 16)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)    #(N, 5776, 4)
        if debugmode==1:
            print("l_conv4_3",l_conv4_3.size())
        assert l_conv4_3.size(1) == 5776

        l_conv7 = self.conv7_loc(conv7_out)    #(N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()    #(N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)    #(N, 2166, 4)
        if debugmode==1:
            print("l_conv7",l_conv7.size())
        assert l_conv7.size(1) == 2166

        l_conv8_2 = self.conv8_2_loc(conv8_2_out)    #(N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()    #(N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)    #(N, 600, 4)
        if debugmode==1:
            print("l_conv8_2",l_conv8_2.size())

        l_conv9_2= self.conv9_2_loc(conv9_2_out)    #(N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()    #(N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)    #(N, 150, 4)
        if debugmode==1:
            print("l_conv9_2",l_conv9_2.size())
        assert l_conv9_2.size(1) == 150

        l_conv10_2 = self.conv10_2_loc(conv10_2_out)    #(N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()    #(N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)    #(N, 36, 4)
        if debugmode==1:
            print("l_conv10_2",l_conv10_2.size())
        assert l_conv10_2.size(1) == 36

        l_conv11_2 = self.conv11_2_loc(conv11_2_out)    #(N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()    #(N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)    #(N, 4, 4)
        if debugmode==1:
            print("l_conv11_2",l_conv11_2.size())
        assert l_conv11_2.size(1) == 4

        #Predict Class in loc boxes -- Predictions for the groups of "Anchor Box" at different scale
        c_conv4_3 = self.conv4_3_cls(conv4_3_out)    #(N, 4*classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()    #(N, 38, 38, 4*classes)
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.num_classes)    #(N, 5776, classes )
        if debugmode==1:
            print("c_conv4_3",c_conv4_3.size())
        assert c_conv4_3.size(1) == 5776

        c_conv7 = self.conv7_cls(conv7_out)    #(N, 6*classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()    #(N, 19, 19, 6*classes)
        c_conv7 = c_conv7.view(batch_size, -1, self.num_classes)   #(N, 2166, classes)
        if debugmode==1:
            print("c_conv7",c_conv7.size())
        assert c_conv7.size(1) == 2166

        c_conv8_2 = self.conv8_2_cls(conv8_2_out)    #(N, 6*clases, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()    #(N, 10, 10, 6*classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.num_classes)    #(N, 600, classes)
        if debugmode==1:
            print("c_conv8_2",c_conv8_2.size())
        assert c_conv8_2.size(1) == 600

        c_conv9_2 = self.conv9_2_cls(conv9_2_out)    #(N, 6*classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()    #(N, 5, 5, 6*classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.num_classes)    #(N, 150, classes)
        if debugmode==1:
            print("c_conv9_2",c_conv9_2.size())
        assert c_conv9_2.size(1) == 150

        c_conv10_2 = self.conv10_2_cls(conv10_2_out)    #(N, 4*classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()    #(N, 3, 3, 4*classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.num_classes)    #(N, 36, classes)
        if debugmode==1:
            print("c_conv10_2",c_conv10_2.size())
        assert c_conv10_2.size(1) == 36

        c_conv11_2 = self.conv11_2_cls(conv11_2_out)    #(N, 4*classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()    #(N, 1, 1, 4*classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.num_classes)    #(N, 4, classes)
        if debugmode==1:
            print("c_conv11_2",c_conv11_2.size())
        assert c_conv11_2.size(1) == 4

        locs_pred = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2,
                               l_conv10_2, l_conv11_2], dim= 1)    #(N, 8732, 4)
        assert locs_pred.size(0) == batch_size
        assert locs_pred.size(1) == 8732
        assert locs_pred.size(2) == 4

        cls_pred = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2,
                              c_conv10_2, c_conv11_2], dim= 1)    #(N, 8732, classes)
        assert cls_pred.size(0) == batch_size
        assert cls_pred.size(1) == 8732
        assert cls_pred.size(2) == self.num_classes

        return segm, depth, locs_pred, cls_pred

    def create_default_boxes(self):
        """ Create 8732 default boxes in center-coordinate (8732, 4) """

        fmap_wh = {"conv4_3": 38, "conv7": 19, "conv8_2": 10, "conv9_2": 5,
                   "conv10_2": 3, "conv11_2": 1}

        #6 scales
        scales = {"conv4_3": 0.1, "conv7": 0.2, "conv8_2": 0.375,
                  "conv9_2": 0.55, "conv10_2": 0.725, "conv11_2": 0.9}

        #different number of aspect ratios for the different scales
        aspect_ratios= {"conv4_3": [1., 2., 0.5], "conv7": [1., 2., 3., 0.5, 0.3333],
                        "conv8_2": [1., 2., 3., 0.5, 0.3333],
                        "conv9_2": [1., 2., 3., 0.5, 0.3333],
                        "conv10_2": [1., 2., 0.5], "conv11_2": [1., 2., 0.5]}

        fmaps = list(fmap_wh.keys())

        default_boxes = []
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_wh[fmap]):
                for j in range(fmap_wh[fmap]):
                    cx = (j + 0.5) / fmap_wh[fmap]
                    cy = (i + 0.5) / fmap_wh[fmap]

                    for ratio in aspect_ratios[fmap]:
                        default_boxes.append([cx, cy, scales[fmap]* math.sqrt(ratio),
                                              scales[fmap]/math.sqrt(ratio)]) #(cx, cy, w, h)

                        if ratio == 1:
                            try:
                                add_scale = math.sqrt(scales[fmap]*scales[fmaps[k+1]])
                            except IndexError:
                                #for the last feature map
                                add_scale = 1.
                            default_boxes.append([cx, cy, add_scale, add_scale])

        default_boxes = torch.FloatTensor(default_boxes).to(device) #(8732, 4)
        default_boxes.clamp_(0, 1)
        assert default_boxes.size(0) == 8732
        assert default_boxes.size(1) == 4
        return default_boxes

    def detect(self, locs_pred, cls_pred, min_score, max_overlap, top_k):
        """ Detect objects, perform NMS on boxes that are above a minimum threshold.
            locs_pred: Pred location, a tensor of dimensions (N, 8732, 4)
            cls_pred: Pred class scores for each of the encoded boxes, a tensor fo dimensions (N, 8732, n_classes)
            min_score: min threshold
            max_overlap: maximum overlap two boxes can have
            top_k: if there are lot of resulting detection across all classes, keep only the top 'k'

            Out: detection list: boxes, labels, score
        """
        batch_size = locs_pred.size(0)                 #N
        n_default_boxes = self.default_boxes.size(0)   #8732
        cls_pred = F.softmax(cls_pred, dim= 2)         #(N, 8732, n_classes)
        assert n_default_boxes == locs_pred.size(1) == cls_pred.size(1)

        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        for i in range(batch_size):
            # Decode object
            decoded_locs = cxcy_to_xy(decode_bboxes(locs_pred[i], self.default_boxes)) #(8732, 4)

            image_boxes = []
            image_labels = []
            image_scores = []

            max_scores, best_label = cls_pred[i].max(dim= 1)

            #Check for each class
            for c in range(1, self.num_classes):
                class_scores = cls_pred[i][:, c]                          # 8732
                score_above_min_score = class_scores > min_score
                n_above_min_score = score_above_min_score.sum().item()

                if n_above_min_score == 0:
                    continue

                # Filter out results that are above the min score
                class_scores = class_scores[score_above_min_score]       # <=8732
                class_decoded_locs = decoded_locs[score_above_min_score] # <=8732

                # Sort pred boxes and socores by scores
                class_scores, sort_id = class_scores.sort(dim= 0, descending= True)
                class_decoded_locs = class_decoded_locs[sort_id]

                # Find overlap between pred locs
                overlap = find_IoU(class_decoded_locs, class_decoded_locs)

                # Apply NMS
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device)

                for box_id in range(class_decoded_locs.size(0)):
                    if suppress[box_id] == 1:
                        continue
                    condition = overlap[box_id] > max_overlap
                    condition = condition.to(device)
                    suppress = torch.max(suppress, condition)

                    suppress[box_id] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[torch.logical_not(suppress)])
                image_labels.append(torch.LongTensor((torch.logical_not(suppress)).sum().item() * [c]).to(device))
                image_scores.append(class_scores[torch.logical_not(suppress)])

            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concat into single tensors
            image_boxes = torch.cat(image_boxes, dim= 0)    #(n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_index = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_index][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_index][:top_k]  # (top_k)

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores

import matplotlib.cm as cm
import matplotlib.colors as co
from PIL import Image, ImageDraw, ImageFont
import torch

def depth_to_rgb(depth):
    """Convert a depth map to a colored RGB image using plasma colormap."""
    normalizer = co.Normalize(vmin=0, vmax=80)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

def drawboxes(image, detect_boxes, detect_labels, detect_scores, label_color_map=None, suppress=None):
    """Draw bounding boxes with labels and confidence scores on the image."""
    try:
        draw = ImageDraw.Draw(image)
    except:
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    if label_color_map is None:
        # Create default color map
        unique_labels = set(detect_labels)
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#888888", "#FFA500", "#800080"]
        label_color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    for i in range(detect_boxes.size(0)):
        if suppress is not None and detect_labels[i] in suppress:
            continue

        box_location = detect_boxes[i].tolist()
        label = detect_labels[i]
        score = detect_scores[i]

        draw.rectangle(xy=box_location, outline=label_color_map[label], width=2)
        text = f"{label} - {score * 100:.1f}%"
        text_size = draw.textsize(text, font=font)
        text_location = [box_location[0], box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0], box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[label])
        draw.text(xy=text_location, text=text, fill='white', font=font)

    return np.array(image)
