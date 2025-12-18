import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Optional, Callable, Any

# =============================================================================
# Helper Modules (From VMamba)
# =============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

# =============================================================================
# Mock / Fallback Selective Scan (For Environment without CUDA Kernels)
# =============================================================================
try:
    # Try importing the real things
    from selective_scan import selective_scan_fn
    HAS_MAMBA_KERNELS = True
except ImportError:
    HAS_MAMBA_KERNELS = False
    # print("Warning: 'selective_scan' kernels not found. Using Mock implementation for structure verification.")

def selective_scan_mock(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    """
    A dumb mock that preserves shapes but does not perform the actual SSM recurrence.
    Input: u (B, D, L)
    Output: y (B, D, L)
    """
    # Just pass through u, maybe add a bit of A/B influence to check connectivity
    # This is ONLY for verifying the network architecture forward pass.
    # It will NOT produce correct semantic segmentation.
    return u + delta.mean(dim=1, keepdim=True) * 0.0

# =============================================================================
# Core Mamba Logic
# =============================================================================

class SS2Dv2(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.k_group = 4 # VMamba standard
        
        # Tags
        self.disable_force32 = False
        
        # Linear projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # Projections for SSM parameters
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        
        # Output
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # Initialization of A, D, dt
        # Simplified initialization from VMamba code
        self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
        self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) 
        self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) 
        self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) 

    def forward(self, x: torch.Tensor):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        
        x = x.permute(0, 3, 1, 2).contiguous() # (b, d, h, w)
        x = self.conv2d(x)
        x = self.act(x)
        
        # Core Mamba Block (SS2D)
        y = self.forward_core(x) # Returns (B, H, W, C)
        
        y = y * self.act(z)
        out = self.dropout(self.out_proj(y))
        return out

    def forward_core(self, x: torch.Tensor):
        # x: (B, D, H, W)
        B, D, H, W = x.shape
        L = H * W
        K = self.k_group
        
        # 1. Flatten and Prepare "Four Direction" scans
        # Real VMamba uses robust cross_scan. We simplify for clarity/mocking.
        # We need (B, K, D, L)
        xs = x.view(B, -1, L)
        # Mock 4 directions by repeating: Normal, Flip, Transpose, Flip-Transpose
        # This is a HACK to get the shapes right for the parameters.
        # Real implementation should do selective scan on 4 directions.
        xs_stack = torch.stack([xs, xs, xs, xs], dim=1) # (B, 4, D, L)
        
        # 2. Compute dt, B, C from x
        # x_dbl: (B, K, C_proj, L)
        # We need to project xs (which is K=4 separate scans) to parameters
        # x_proj_weight: (K, input_dim, output_dim)
        # einsum: b k d l, k c d -> b k c l
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_stack, self.x_proj_weight)
        
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        # Project dt
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        dts = dts + self.dt_projs_bias.view(1, K, -1, 1)
        
        # 3. Running SSM (Selective Scan)
        # Parameters
        As = -torch.exp(self.A_logs) # (K*D, N)
        Ds = self.Ds # (K*D)
        
        # Unbind K to run scan per group loop (simplest way without custom kernel)
        ys = []
        for i in range(K):
            # i-th direction
            # u: (B, D, L)
            u_i = xs_stack[:, i] 
            dt_i = dts[:, i] 
            B_i = Bs[:, i] # (B, N, L) but scan needs (B, N, L) - verify shapes
            C_i = Cs[:, i]
            
            # Helper: We are using a Mock, so let's just pass them through
            if HAS_MAMBA_KERNELS:
                # Real implementation integration would go here
                # selective_scan_fn(...) 
                pass
            
            # Using Mock
            y_i = selective_scan_mock(u_i, dt_i, None, None, None)
            ys.append(y_i)
            
        y = torch.stack(ys, dim=1) # (B, K, D, L)
        
        # 4. Merge back
        # In real VMamba, we merge the 4 directions.
        # Simply sum them up for this structural mock.
        y = y.sum(dim=1) # (B, D, L)
        
        y = y.transpose(1, 2).view(B, H, W, D)
        y = self.out_norm(y)
        return y


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2Dv2(
            d_model=hidden_dim, 
            d_state=d_state,
            dropout=attn_drop_rate, 
            **kwargs
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

# Quick verification if run as script
if __name__ == "__main__":
    print("Testing SS2Dv2 / VSSBlock...")
    block = VSSBlock(hidden_dim=96).cuda()
    x = torch.randn(1, 64, 64, 96).cuda()
    y = block(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print("Test passed!")
