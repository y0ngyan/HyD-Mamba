import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttention(nn.Module):
    """
    Standard Multi-Head Self Attention on a selected subset of tokens.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B, N, C) where N is the number of selected tokens (k)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DuSABlock(nn.Module):
    """
    Dual-Stage Sparse Attention Block.
    1. Estimate importance map (or use Depth Edge).
    2. Select Top-K pixels.
    3. Apply Attention on Top-K.
    4. Scatter + Residual.
    """
    def __init__(self, dim, num_heads=8, topk_ratio=0.1, importance_ratio=4):
        super().__init__()
        self.dim = dim
        self.topk_ratio = topk_ratio
        
        # Importance estimator (if no external guide provided, or to refine it)
        # Squeeze-and-Excitation like or simple Conv
        self.importance_net = nn.Sequential(
            nn.Conv2d(dim, dim // importance_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(dim // importance_ratio, 1, 1),
            nn.Sigmoid()
        )
        
        self.attn = SparseAttention(dim, num_heads=num_heads)
        self.gamma = nn.Parameter(torch.zeros(1)) # Zero init for residual

    def forward(self, x, guide_map=None):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        k = int(N * self.topk_ratio)
        
        # 1. Calculate Importance
        imp = self.importance_net(x) # (B, 1, H, W)
        if guide_map is not None:
            # guide_map should be (B, 1, H, W), e.g. edge map from depth
            # Resize guide if needed
            if guide_map.shape[-2:] != (H, W):
                guide_map = F.interpolate(guide_map, size=(H, W), mode='bilinear', align_corners=False)
            imp = imp + guide_map
            
        # 2. Select Top-K
        imp_flat = imp.view(B, -1) # (B, N)
        topk_vals, topk_inds = torch.topk(imp_flat, k, dim=1) # (B, k)
        
        # Gather features
        # x_flat: (B, C, N) -> permute -> (B, N, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1) 
        
        # Gather logic: we need to expand indices to (B, k, C)
        batch_inds = torch.arange(B, device=x.device).unsqueeze(1).expand(B, k)
        # Simple gathering is tricky in batch without advanced indexing or gather
        # x_selected = x_flat[batch_inds, topk_inds, :] # This works in PyTorch
        
        # Safe gathering using torch.gather
        # indices for gather must have same dims as input except dim to gather
        # We want to gather along dim 1 (N tokens).
        # topk_inds: (B, k) -> expand to (B, k, C)
        gather_inds = topk_inds.unsqueeze(-1).expand(-1, -1, C)
        x_selected = torch.gather(x_flat, 1, gather_inds) # (B, k, C)
        
        # 3. Sparse Attention
        x_attn = self.attn(x_selected) # (B, k, C)
        
        # 4. Scatter / Fuse back
        # We create a zero canvas or residual on x
        # Efficient way: add to x_flat at indices
        # scatter_add_(dim, index, src)
        # index must be same size as src.
        
        # x_out = x_flat.clone() # Clone is expensive? 
        # Actually we want residual: out = x + gamma * sparse_update
        # Sparse update is 0 everywhere except topk.
        
        sparse_update = torch.zeros_like(x_flat)
        sparse_update.scatter_add_(1, gather_inds, x_attn)
        
        # Scale and add
        out = x_flat + self.gamma * sparse_update
        out = out.permute(0, 2, 1).view(B, C, H, W)
        
        return out

if __name__ == "__main__":
    print("Testing DuSABlock...")
    block = DuSABlock(dim=64, topk_ratio=0.1).cuda()
    x = torch.randn(1, 64, 32, 32).cuda()
    guide = torch.rand(1, 1, 32, 32).cuda()
    y = block(x, guide)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print("Test passed!")
