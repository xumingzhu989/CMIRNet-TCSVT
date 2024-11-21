import torch
from torch import nn
from torch.nn import functional as F

class TGMM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(TGMM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                        #  nn.GELU(),
                                         nn.ReLU(),
                                         nn.Dropout(dropout)
                                         )

        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=dim,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),
                                        # nn.GELU(),
                                        nn.ReLU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask, emb):
        # input x shape: (B, H*W, dim)
        # l B, C, N
        B, C, H, W = x.shape
        vis = self.vis_project(x.view(B, C, -1))  # (B, dim, H*W)

        lang, l_new, word_new = self.image_lang_att(x, l, l_mask, emb)  # (B, dim, 1)
        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.view(B, C, H, W)

        return mm, l_new, word_new


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        self.f_value2 = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        # Out projection
        self.W1 = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )
        self.W2 = nn.Sequential(
            nn.Conv1d(self.value_channels, self.value_channels, kernel_size=1, stride=1),
        )
        self.W3 = nn.Sequential(
            nn.Conv1d(self.value_channels, self.value_channels, kernel_size=1, stride=1),
        )
        self.project_emb = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)

    def forward(self, x, l, l_mask, emb):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        HW = H * W
        # x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        value2 = self.f_value2(l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        value2 = value2 * l_mask
        n_l = value2.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)
        value2 = value2.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        att_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4 * att_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        pool_map = nn.AdaptiveAvgPool2d([1, n_l])(sim_map)  # (B,num_heads,1,N_l)

        l_new = (pool_map * value2).reshape(B, self.value_channels, n_l)  # (B,C,N_l)

        l_new_2 = self.W2(l_new)

        # value = self.f_value(l_new_2)  # (B, self.value_channels, N_l)
        # value = value * l_mask  # (B, self.value_channels, N_l)
        # value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)

        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W1(out)  # (B, value_channels, HW)

        # out = out.view(B, C, H, W)

        emb = emb.unsqueeze(-1)
        emb = self.project_emb(emb)
        l_new = nn.AdaptiveAvgPool2d([self.value_channels, 1])(l_new) * self.gamma + emb  # (B,C,1)
        l_out = self.W3(l_new) + emb

        return out, l_out, l_new_2


