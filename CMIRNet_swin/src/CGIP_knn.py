from torch.nn.init import kaiming_normal_, constant_
# from .model_util import *
import torch
from torch import nn
import torch.nn.functional as F


class IGR_MH(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim / self.num_heads) ** -0.5

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            # nn.GELU()
            nn.ReLU(),
        )
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Vx):
        Vx = self.conv(Vx)

        B, C, H, W = Vx.shape
        N_node = H * W

        query, key, value = self.qkv(Vx).reshape(B, self.num_heads, C // self.num_heads * 3, N_node).chunk(3, dim=2)  # B,head,C_head,N_node

        attn = query.transpose(-1, -2) @ key * self.scale
        attn = self.softmax(attn)  # B,head,N_node,N_node

        out = (value @ attn.transpose(-1, -2))  # B,head,C_head,N_node
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return out



class CGI_MH(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim / self.num_heads) ** -0.5

        # self.conv_V = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim),
        #     nn.GELU()
        # )

        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.kv = nn.Conv1d(dim, dim * 2, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Vx, Tx):
        # Vx = self.conv_V(Vx)
        B,C,H,W = Vx.shape
        N = H*W
        _,_,T = Tx.shape
        
        V_query = self.q(Vx).reshape(B, self.num_heads, C // self.num_heads, N)  # B,head,C_head,N
        T_key, T_value = self.kv(Tx).reshape(B, self.num_heads, C // self.num_heads * 2, T).chunk(2, dim=2)  # B,head,C_head,T

        attn = V_query.transpose(-1, -2) @ T_key * self.scale  # B,head,N,T
        attn = self.softmax(attn)  # B,head,N,T

        out = (T_value @ attn.transpose(-1, -2))  # B,head,C_head,N
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return out#+Vx
        

class CGIP_knn(nn.Module):
    def __init__(self, dim=256, num_heads=1):
        super().__init__()
        self.conv_V = nn.Sequential(
            nn.Conv2d(1024, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.conv_T = nn.Sequential(
            nn.Conv1d(768, dim, kernel_size=1, stride=1),
            nn.InstanceNorm1d(dim),
            nn.ReLU()
        )
        self.CGI = MH_DyGraphConv2d(dim, num_heads=num_heads)
        self.IGR = IGR_MH(dim=dim, num_heads=num_heads)
        
    def forward(self, Vx, Tx):
        Vx = self.conv_V(Vx)
        # l, l1, l2, l3, l4, laspp = Tx
        # L = self.conv_T(l) + l1 + l2 + l3 + l4 + laspp
        l, _, l2, l3, l4, laspp = Tx
        L = self.conv_T(l) + l2 + l3 + l4 + laspp
        out = self.CGI(Vx, L)
        out = self.IGR(out)
        return out + Vx


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, x_num_points, num_dims)
        y: tensor (batch_size, y_num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, x_num_points, y_num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def xy_dense_knn_matrix(x, y, k=16):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, x_num_points, 1)
        y: (batch_size, num_dims, y_num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):  # 如果需要膨胀，则输入的edge_index中的维度k应该是所有节点n，然后从所有节点中选出膨胀的邻居子集来。
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.k = k

    def forward(self, edge_index):
        edge_index = edge_index[:, :, :, ::self.dilation]  # 间隔self.dilation去取邻居，邻居数量变为k/self.dilation个
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k, dilation):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.k = k
        self._dilated = DenseDilated(k, dilation)

    def forward(self, x, y):
        #### normalize
        x = F.normalize(x, p=2.0, dim=1)
        y = F.normalize(y, p=2.0, dim=1)
        ####
        edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation)
        return 0, self._dilated(edge_index)  # (2, batch_size, num_points, k)
    
class MH_DyGraphConv2d(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        # self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.graphconv = nn.ModuleList([DyGraphConv2d(dim // num_heads, dim // num_heads, kernel_size=9, dilation=1, conv='edge', act='gelu') for i in range(num_heads)])
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.linear = nn.Linear(dim // num_heads, 3 * dim // num_heads)

    def forward(self, x, y):  # (B, C, H, W) head=4  x: (1, 96, 8, 8)
        B, C, H, W = x.shape
        B, C, T = y.shape
        x = x.chunk(self.num_heads, dim=1)
        y = y.chunk(self.num_heads, dim=1)
        x = [self.graphconv[i](x[i], y[i]).unsqueeze(1) for i in range(self.num_heads)]
        x = torch.cat(x, dim=1)  # (B, head, C/head, H, W)  if head=4  (1, 4, 24, 8, 8)
        f = self.global_max_pool(x.reshape(-1, self.dim // self.num_heads, H, W)).squeeze(-1).squeeze(-1).reshape(B,
                                                                                                             self.num_heads,
                                                                                                             -1)  # (B, head, C/head)  if head=4  (1, 4, 24)
        f_k, f_q, f_v = self.linear(f).transpose(1, 2).chunk(3, dim=-2)  # if head=4     (1, 24, 4)
        att = torch.softmax(f_q.transpose(-1, -2) @ f_k, dim=-2)
        wt = (f_v @ att).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # if head=4    (1, 4, 24)
        x = (x * wt).reshape(B, C, H, W)
        return x  # (1, 96, 8, 8)

# class MH_DyGraphConv2d(nn.Module):
#     def __init__(self, dim, num_heads=4):
#         super().__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU()
#         )
#         self.graphconv = nn.ModuleList([DyGraphConv2d(dim // num_heads, dim // num_heads, kernel_size=9, dilation=1, conv='edge', act='gelu') for i in range(num_heads)])

#     def forward(self, x, y):  # (B, C, H, W) head=4  x: (1, 96, 8, 8)
#         B, C, H, W = x.shape
#         B, C, T = y.shape
#         x = x.chunk(self.num_heads, dim=1)
#         y = y.chunk(self.num_heads, dim=1)
#         x = [self.graphconv[i](x[i], y[i]) for i in range(self.num_heads)]
#         x = torch.cat(x, dim=1)  # (B, head, C/head, H, W)  if head=4  (1, 4, 24, 8, 8)
#         x = self.conv(x)  # B, C, H, W
#         return x  # (1, 96, 8, 8)


class DyGraphConv2d(nn.Module):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu'):
        super(DyGraphConv2d, self).__init__()
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation)  # 求最近邻
        self.homoGR_feat = GraphConv2d(in_channels, out_channels, conv='edge', act=act, norm=None, bias=True)

    def forward(self, x, y):
        B, C, H, W = x.shape
        B, C, T = y.shape
        x = x.reshape(B, C, -1, 1).contiguous()  # bc(hw)1
        y = y.unsqueeze(-1)  # bcT1
        _, edge_index_fn = self.dilated_knn_graph(x, y)  # 2bnk
        x_nn_feat = self.homoGR_feat(x, edge_index_fn, y)  # B,C,N,1
        x = x_nn_feat
        return x.reshape(B, -1, H, W).contiguous()  # B,C,H,W

class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
    def forward(self, x, edge_index, y):
        return self.gconv(x, edge_index, y)  # B,C,N,1
    
#节点聚合：将节点原特征与所有邻居节点特征差异在通道维度进行拼接，然后在通道维度聚合生成所有K个邻居对当前节点的贡献，然后选出贡献最大的节点特征作为更新的特征
class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU())

    def forward(self, x, edge_index, y=None):  # x is B,C,N,1
        x_i = batched_index_select(x, edge_index[1])  # edge_index[1]保存的是行号，x_i是按行提取的特征（B,C,N,K），对于N中的第i行的第i个节点，其有K列，即第i个节点特征有K份；
        x_j = batched_index_select(y, edge_index[0])  # edge_index[0]保存的是列号，x_j是按列提取的特征（B,C,N,K），对于N中的第j行的第j个节点，其有K列，即第i个节点的K个邻居节点的特征；
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)  # B,C,N,K max→ B,C,N,1
        return max_value  # B,C,N,1
    
    
def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base  # B,N,k
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]  # BNK,C
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()  # B,C,N,K
    return feature