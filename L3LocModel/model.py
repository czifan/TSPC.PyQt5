from torchvision.models import *
from torchvision.models.shufflenetv2 import *
import torch 
import torch.nn as nn
import os 
from torch.nn import functional as F
import numpy as np

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class FeedForward(nn.Module):
    def __init__(self, dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, feat_dim, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(feat_dim, dim)
        self.proj_k = nn.Linear(feat_dim, dim)
        self.proj_v = nn.Linear(feat_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = torch.cosine_similarity(q[:, :, :, None, :], q[:, :, None, :, :], dim=-1)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        attn_map = F.softmax(scores, dim=-1)
        scores = self.drop(attn_map)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = attn_map
        return h

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, feat_x, mask):
        attn_x = self.attn(feat_x, mask)
        attn_x = self.attn_norm(attn_x)
        attn_x = attn_x + feat_x 
        mlp_x  = self.mlp(attn_x)
        mlp_x  = self.mlp_norm(mlp_x)
        mlp_x  = self.drop(mlp_x)
        out_x  = mlp_x + attn_x
        return out_x

class SliceTransformer(nn.Module):
    def __init__(self,
                in_dim=512,
                num_head=8,
                dropout=0.5,
                num_attn=2,
                center_id=2):
        super().__init__()
        self.attn_layer_lst = nn.ModuleList([
            Block(in_dim, num_head, dropout) for _ in range(num_attn)
        ])
        self.center_id = center_id
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)

    def forward(self, x, pe=None):
        # x: (B, T, C)
        B, N, C = x.shape
        mask = torch.ones(B, N).float().to(x.device)
        for attn_layer in self.attn_layer_lst:
            if pe is not None:
                x = x + pe
            x = attn_layer(x, mask)
        return x[:, self.center_id] # (B, C)

class L3LocModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        backbone = eval(args.backbone)(pretrained=args.pretrained)
        if "shufflenet" in args.backbone:
            backbone.conv1[0] = nn.Conv2d(1, backbone.conv1[0].out_channels,
                                      kernel_size=backbone.conv1[0].kernel_size,
                                      stride=backbone.conv1[0].stride,
                                      bias=backbone.conv1[0].bias)
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif "resnet" in args.backbone:
            backbone.conv1 = nn.Conv2d(1, backbone.conv1.out_channels,
                                      kernel_size=backbone.conv1.kernel_size,
                                      stride=backbone.conv1.stride,
                                      bias=backbone.conv1.bias)
            self.pool = None
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        # self.st = SliceTransformer(in_dim=args.d_model, dropout=args.dropout, center_id=args.N_neighbor)
        
        self.classifier = nn.Sequential(
            #nn.LayerNorm(args.d_model),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model*(1+2*args.N_neighbor), args.d_model),
            nn.Linear(args.d_model, 1)
        )
        self.N_neighbor = args.N_neighbor
        self.args = args

    def forward(self, x, N_lst=None):
        # x: (B, N, 3, 96, 96)
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        cnn_feat = self.cnn(x)
        if self.pool:
            cnn_feat = self.pool(cnn_feat)
        cnn_feat = cnn_feat.view(B, N, 1, -1) # (B, N, 1, 512)
        
        feat = []
        for n in range(-self.N_neighbor, self.N_neighbor+1): # (-2, -1, 0, 1, 2)
            if n <= 0: 
                tmp = cnn_feat[:, abs(n):, ...]
                tmp = torch.cat([tmp, torch.zeros(B, abs(n), *cnn_feat.shape[-2:]).float().to(cnn_feat.device)], dim=1)
            else: 
                tmp = cnn_feat[:, :-n, ...]
                tmp = torch.cat([torch.zeros(B, n, *cnn_feat.shape[-2:]).float().to(cnn_feat.device), tmp], dim=1)
            feat.append(tmp)
        feat = torch.cat(feat, dim=2) # (B, N, 1+2*N_neighbor, 512)
        feat = feat.view(B, N, -1)
        pred = self.classifier(feat).squeeze(dim=2)
        #feat = feat.view(B*N, -1, feat.shape[-1])
        #st_feat = self.st(feat).view(B, N, feat.shape[-1]) # (B, N, 512)
        #preb = self.classifier(st_feat).squeeze(dim=2) # (B, N)
        return pred
    
class L3LocModelV1(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        backbone = eval(args.backbone)(pretrained=args.pretrained)
        if "shufflenet" in args.backbone:
            backbone.conv1[0] = nn.Conv2d(1, backbone.conv1[0].out_channels,
                                      kernel_size=backbone.conv1[0].kernel_size,
                                      stride=backbone.conv1[0].stride,
                                      bias=backbone.conv1[0].bias)
            self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        elif "resnet" in args.backbone:
            backbone.conv1 = nn.Conv2d(1, backbone.conv1.out_channels,
                                      kernel_size=backbone.conv1.kernel_size,
                                      stride=backbone.conv1.stride,
                                      bias=backbone.conv1.bias)
            self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        
        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model, args.d_model),
            nn.Linear(args.d_model, 1)
        )
        self.args = args

    def forward(self, x, N_lst):
        # x: (B, 1, 96, 96)
        B, C, H, W = x.shape
        cnn_feat = self.cnn(x) # (B, C, H, W)
        cnn_feat = torch.max(cnn_feat, dim=-1).values # (B, C, H)
        cnn_feat = cnn_feat.permute(0, 2, 1).contiguous() # (B, H, C)
        
        feat = self.classifier(cnn_feat) # (B, H, 1)
        preb = []
        for f, N in zip(feat, N_lst):
            preb.append(F.interpolate(f.view(1, 1, 1, -1), size=(1, N), mode="bilinear").view(-1))
        pred = torch.stack(preb, dim=0)
        return pred
    
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--N_neighbor', type=int, default=2)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = get_args()
    model = L3LocModel(args).cuda()
    x = torch.zeros(2, 10, 1, 96, 96).float().cuda()
    print(model(x).shape) 
        
        
        
        
        