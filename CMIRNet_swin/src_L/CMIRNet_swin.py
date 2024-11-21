import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from .CGIP_knn import CGIP_knn
from .CMAE_512 import CMAE_512

class CMIRNet_swin(nn.Module):
    def __init__(self, Vis_backbone_TGMM, num_classes, args):
        super(CMIRNet_swin, self).__init__()
        self.Cmi = 512
        self.Vis_backbone_TGMM = Vis_backbone_TGMM
        self.gcn = CGIP_knn(dim=self.Cmi, num_heads=8)
        self.cmae = CMAE_512(self.Cmi)

        self.conv_2_1 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_3 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_4 = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)
        self.conv_2_aspp = nn.Conv2d(self.Cmi, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, word_ori,  l_mask, emb_ori):  # word_ori 是原始词向量，emb_ori是原始句向量
        input_shape = x.shape[-2:]
        result = OrderedDict()
        x_cmae, atts, lans, words, x_aspp = self.Vis_backbone_TGMM(x, word_ori, l_mask, emb_ori)
        x_c1, x_c2, x_c3, x_c4, x_c_aspp = x_cmae  # 输入到CMAE
        lan1, lan2, lan3, lan4, lan_aspp = lans  # 新的句向量，输入到CMAE
        word1, word2, word3, word4, word_aspp = words # 新的词向量，输入到CGIP

        x_gcn = self.gcn(x_aspp, tuple([word_ori,word1,word2,word3, word4, word_aspp]))

        f1, f2, f3, f4, faspp = self.cmae(tuple([x_c1, x_c2, x_c3, x_c4, x_c_aspp, x_gcn]), tuple([lan1, lan2, lan3, lan4, lan_aspp]))

        x1 = self.conv_2_1(f1)
        x2 = self.conv_2_2(f2)
        x3 = self.conv_2_3(f3)
        x4 = self.conv_2_4(f4)
        xaspp = self.conv_2_aspp(faspp)

        x1 = F.interpolate(x1, size=input_shape, mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=input_shape, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=input_shape, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=input_shape, mode='bilinear', align_corners=False)
        xaspp = F.interpolate(xaspp, size=input_shape, mode='bilinear', align_corners=False)

        result["out"] = x1
        result["out2"] = x2
        result["out3"] = x3
        result["out4"] = x4
        result["outaspp"] = xaspp

        return result, x, emb_ori