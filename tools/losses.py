import torch
import torch.nn as nn
import torch.nn.functional as F

class OhemCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining Cross Entropy Loss.
    """
    def __init__(self, thres=0.7, min_kept=100000, ignore_index=255):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thres
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        pixel_losses = self.criterion(pred, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_index
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_index] = 0
        pred = F.softmax(pred, dim=1)
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        if not mask.any():
            return pixel_losses.new_zeros(1, requires_grad=True).mean()

        pixel_losses = pixel_losses[mask]
        pred = pred.contiguous().view(-1,)[mask]
        
        # Sort and pick threshold
        pred, ind = pred.sort()
        
        num_kept = min(self.min_kept, pred.numel())
        if num_kept > 0:
            min_value = pred[num_kept - 1]
            threshold = max(min_value, self.thresh)
        else:
            threshold = self.thresh

        pixel_losses = pixel_losses[ind]
        pixel_losses = pixel_losses[pred < threshold]
        
        if pixel_losses.numel() == 0:
            # Fallback if no pixels match (e.g. all pixels are correct)
            return pred.new_zeros(1, requires_grad=True).mean()
            
        return pixel_losses.mean()

class DetailAggregateLoss(nn.Module):
    """
    Encourages network to pay attention to details by using Depth Gradient as a weight map.
    If depth gradient is high (edge), importance is higher.
    """
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        
    def get_gradient(self, depth):
        # Sobel operator or simple diff
        # depth: (B, 1, H, W)
        b, c, h, w = depth.shape
        dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
        dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
        
        # Pad to keep size
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        
        grad = dy + dx
        return grad
        
    def forward(self, pred, target, depth):
        # pred: (B, C, H, W)
        # target: (B, H, W)
        # depth: (B, 1, H, W)
        
        loss = self.ce(pred, target) # (B, H, W)
        
        # Calculate weight map from depth
        grad = self.get_gradient(depth).squeeze(1) # (B, H, W)
        # Normalize grad to 0-1 or 1-2 range
        weight = 1.0 + torch.tanh(grad) # 1.0 (flat) to 2.0 (edge)
        
        loss = (loss * weight).mean()
        return loss

if __name__ == "__main__":
    print("Testing Losses...")
    pred = torch.randn(2, 19, 128, 128)
    target = torch.randint(0, 19, (2, 128, 128))
    depth = torch.rand(2, 1, 128, 128)
    
    ohem = OhemCrossEntropyLoss(min_kept=100)
    l1 = ohem(pred, target)
    print(f"OHEM Loss: {l1.item()}")
    
    detail = DetailAggregateLoss()
    l2 = detail(pred, target, depth)
    print(f"Detail Loss: {l2.item()}")
