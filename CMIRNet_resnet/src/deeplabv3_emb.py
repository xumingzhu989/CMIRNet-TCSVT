import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from .ASPP import ASPP_v4
from .CGIP_knn import CGIP_knn
from .CMAE_512 import CMAE_512
from .TGMM import TGMM

__all__ = ["DeepLabV3Emb_512"]


class DeepLabV3Emb_512(nn.Module):
    def __init__(self, backbone, num_classes, args):
        super(DeepLabV3Emb_512, self).__init__()
        self.Cmi = 512
        self.backbone = backbone
        self.layer1 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu)
        self.layer2, self.layer3, self.layer4, self.layer5 = nn.Sequential(self.backbone.maxpool,
                                                                           self.backbone.layer1), self.backbone.layer2, self.backbone.layer3, self.backbone.layer4
        # for n, m in self.layer4.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)
        # for n, m in self.layer5.named_modules():
        #    if 'conv2' in n:
        #        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
        #    elif 'downsample.0' in n:
        #        m.stride = (1, 1)

        self.TGMM3 = TGMM(512, 512, 768, self.Cmi, self.Cmi, num_heads=8)
        self.TGMM4 = TGMM(1024, 1024, 768, self.Cmi, self.Cmi, num_heads=8)
        self.TGMM5 = TGMM(2048, 2048, 768, self.Cmi, self.Cmi, num_heads=8)
        self.TGMM_aspp = TGMM(2048, 2048, 768, self.Cmi, self.Cmi, num_heads=8)

        self.aspp = ASPP_v4(2048, self.Cmi, [3, 6, 9], args)
        self.gcn = CGIP_knn(dim=self.Cmi, num_heads=8)
        self.cmae = CMAE_512(self.Cmi)

        self.conv_2_1 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_3 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_4 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_5 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_aspp = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, word, emb, mask):
        input_shape = x.shape[-2:]
        result = OrderedDict()

        x1 = self.layer1(x)  # 64 * H/2 * W/2
        x2 = self.layer2(x1)  # 256 * H/4 * W/4

        x3 = self.layer3(x2)  # 512 * H/8 * W/8
        res3, l3, word3 = self.TGMM3(x3, word, mask, emb)
        x3 = x3 + res3

        x4 = self.layer4(x3)  # 1024 * H/16 * W/16
        res4, l4, word4 = self.TGMM4(x4, word, mask, emb)
        x4 = x4 + res4

        x5 = self.layer5(x4)  # 2048 * H/32 * W/32
        res5, l5, word5 = self.TGMM5(x5, word, mask, emb)
        x5 = x5 + res5

        xaspp = self.aspp(x5)
        res_aspp, l_aspp, word_aspp = self.TGMM_aspp(xaspp, word, mask, emb)
        xaspp = xaspp + res_aspp

        x_gcn = self.gcn(xaspp, tuple([word, word3, word4, word5, word_aspp]))

        f1, f2, f3, f4, f5, faspp = self.cmae(tuple([x1, x2, res3, res4, res5, res_aspp, x_gcn]), word, mask,
                                              tuple([l3, l4, l5, l_aspp]), emb)

        x1 = self.conv_2_1(f1)
        x2 = self.conv_2_2(f2)
        x3 = self.conv_2_3(f3)
        x4 = self.conv_2_4(f4)
        x5 = self.conv_2_5(f5)
        xaspp = self.conv_2_aspp(faspp)

        x1 = F.interpolate(x1, size=input_shape, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=input_shape, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=input_shape, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=input_shape, mode='bilinear', align_corners=False)
        x5 = F.interpolate(x5, size=input_shape, mode='bilinear', align_corners=False)
        xaspp = F.interpolate(xaspp, size=input_shape, mode='bilinear', align_corners=False)

        result["out"] = x1
        result["out2"] = x2
        result["out3"] = x3
        result["out4"] = x4
        result["out5"] = x5
        result["outaspp"] = xaspp

        return result, x, emb

