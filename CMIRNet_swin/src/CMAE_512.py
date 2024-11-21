import torch
from torch import nn
from torch.nn import functional as F


class CMAE_512(nn.Module):
    def __init__(self,dim=256):
        super(CMAE_512, self).__init__()
        self.dropout = 0.0
        self.Cmi = 256
        self.linear_x_1 = nn.Sequential(
            nn.Conv2d(128, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.linear_x_2 = nn.Sequential(
            nn.Conv2d(256, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.linear_x_3 = nn.Sequential(
            nn.Conv2d(512, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.linear_x_4 = nn.Sequential(
            nn.Conv2d(1024, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.linear_x_aspp = nn.Sequential(
            nn.Conv2d(1024, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.ConvBNRelu_aspp_1= nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ConvBNRelu_aspp_2= nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ConvBNRelu4_1 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ConvBNRelu4_2 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ConvBNRelu3_1 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ConvBNRelu3_2 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ConvBNRelu2_1 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ConvBNRelu2_2 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.ConvBNRelu1_1 = nn.Sequential(
            nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        # self.ConvBNRelu1_2 = nn.Sequential(
        #     nn.Conv2d(dim + dim, dim, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(),
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(),
        #     nn.Dropout(self.dropout)
        # )

        # self.fusion1 = Fusion(dim=dim)
        self.fusion2 = Fusion(dim=dim)
        self.fusion3 = Fusion(dim=dim)
        self.fusion4 = Fusion(dim=dim)
        self.fusionaspp = Fusion(dim=dim)

        self.up1 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, res, lans):
        x1, x2, x3, x4, x_aspp, x_gcn = res
        l1, l2, l3, l4, l_aspp = lans

        x1 = self.linear_x_1(x1)
        x2 = self.linear_x_2(x2)
        x3 = self.linear_x_3(x3)
        x4 = self.linear_x_4(x4)
        x_aspp = self.linear_x_aspp(x_aspp)

        res_aspp = self.ConvBNRelu_aspp_1(torch.cat((x_aspp, x_gcn), dim=1))
        cmae_aspp = self.fusionaspp(res_aspp, l_aspp)
        res_aspp = self.ConvBNRelu_aspp_2(torch.cat((res_aspp, cmae_aspp), dim=1))

        res4 = self.ConvBNRelu4_1(torch.cat((self.up1(res_aspp), x4), dim=1))
        cmae4 = self.fusion4(res4, l4)
        res4 = self.ConvBNRelu4_2(torch.cat((res4, cmae4), dim=1))

        res3 = self.ConvBNRelu3_1(torch.cat((self.up2(res4), x3), dim=1))
        cmae3 = self.fusion3(res3, l3)
        res3 = self.ConvBNRelu3_2(torch.cat((res3, cmae3), dim=1))

        res2 = self.ConvBNRelu2_1(torch.cat((self.up2(res3), x2), dim=1))
        cmae2 = self.fusion2(res2, l2)
        res2 = self.ConvBNRelu2_2(torch.cat((res2, cmae2), dim=1))

        res1 = self.ConvBNRelu1_1(torch.cat((self.up2(res2), x1), dim=1))
        # cmae1 = self.fusion1(res1, l1)
        # res1 = self.ConvBNRelu1_2(torch.cat((res1, cmae1), dim=1))

        return tuple([res1, res2, res3, res4, res_aspp])


class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.csa = CSA(dim)
        self.cca = CCA(dim)

    def forward(self, x, l):
        Csa = self.csa(x, l)
        Cca = self.cca(x, Csa)
        res = Cca.unsqueeze(-1) * x + x

        Csa_ = Csa.permute(0, 2, 1).view(x.shape[0], 1, x.shape[-2], x.shape[-1])

        res = res * Csa_ + x
        return res


class CSA(nn.Module):
    def __init__(self, dim):
        super(CSA, self).__init__()
        self.w1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.InstanceNorm1d(dim), )
        self.w2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1), )
        self.dim = dim

    def forward(self, x, f):
        # x(B,c,H, W)
        # f(B,Cl,1)
        x2 = x.view(x.size(0), x.size(1), -1)  # (B,c,H, W) -> (B,c,HW)

        q = self.w2(f)  # (B, C, 1)
        q = F.normalize(q, dim=1, p=2)

        k = self.w1(x2)
        k = F.normalize(k, dim=1, p=2)
        k = k.permute(0, 2, 1)  # (B,HW,256)

        att = torch.matmul(k, q)  # (B, HW, 1)
        return att


class CCA(nn.Module):
    def __init__(self, dim=256):
        super(CCA, self).__init__()
        self.w1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            nn.InstanceNorm1d(dim), )
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Conv1d(dim, dim // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // 16, dim, 1, bias=False)
        )

    def forward(self, x, f):
        # x(B,c,H, W)
        # f(B,HW,1)
        x = x.view(x.size(0), x.size(1), -1)  # (B,c,H, W) -> (B,c,HW)

        B, _, HW = x.shape

        q = f
        q = F.softmax(q, dim=-2)

        k = self.w1(x)

        att1 = torch.matmul(k, q)  # (B, C, 1)

        # x_pool = nn.AdaptiveMaxPool1d(1)(x)

        att1 = self.mlp(att1)
        # att2 = self.mlp(x_pool)

        att = F.sigmoid(att1)

        return att
