import torch
import torch.nn as nn
import torch.fft
import numpy as np

class DirectionalFilterBank(nn.Module):
    def __init__(self, num_bands=8, device='cpu'):
        super(DirectionalFilterBank, self).__init__()
        self.num_bands = num_bands
        self.device = device
        
    def generate_soft_angular_masks(self, shape, sharpness=10.0):
        """
        Tạo mặt nạ hình nêm với biên mềm (Sigmoid transition) để giảm Ringing artifacts.
        """
        H, W = shape
        y = torch.arange(H, device=self.device) - H // 2
        x = torch.arange(W, device=self.device) - W // 2
        Y, X = torch.meshgrid(y, x, indexing='ij')
        angles = torch.atan2(Y, X) # [-pi, pi]
        
        masks = []
        step = np.pi / self.num_bands
        
        for k in range(self.num_bands):
            # Xác định góc trung tâm của band
            center_angle = k * step - np.pi/2 + step/2
            
            # Tính khoảng cách góc (ngắn nhất trên vòng tròn)
            # Dùng cos để đo độ gần: cos(a-b) càng gần 1 thì càng gần trung tâm band
            # cos(diff) = 1 -> tâm band. cos(diff) < ngưỡng -> ngoài band.
            
            # Góc rộng của band là step. Bán kính góc là step/2.
            # Ta tạo hàm Gaussian hoặc Sigmoid dựa trên khoảng cách tới center_angle
            
            # Cách đơn giản: Distance map
            # diff = min(|a - center|, |a - center + pi|, |a - center - pi|)
            diff = torch.abs(angles - center_angle)
            diff = torch.min(diff, torch.abs(angles - (center_angle + np.pi)))
            diff = torch.min(diff, torch.abs(angles - (center_angle - np.pi)))
            
            # Sigmoid mask: 1 ở trong band (diff < step/2), 0 ở ngoài
            # Sharpness điều chỉnh độ dốc biên
            cutoff = step / 2.0
            mask = torch.sigmoid(sharpness * (cutoff - diff))
            
            masks.append(mask.float())
            
        # Chuẩn hóa để tổng các mask tại mọi điểm = 1 (Partition of Unity)
        mask_sum = torch.stack(masks, dim=0).sum(dim=0) + 1e-8
        masks = [m / mask_sum for m in masks]
            
        return torch.stack(masks)

    def forward(self, x):
        B, C, H, W = x.shape
        fft_x = torch.fft.fftshift(torch.fft.fft2(x))
        
        if not hasattr(self, 'masks') or self.masks.shape[-2:] != (H, W):
            self.masks = self.generate_soft_angular_masks((H, W))
            
        subbands = []
        for k in range(self.num_bands):
            mask = self.masks[k].unsqueeze(0).unsqueeze(0)
            fft_masked = fft_x * mask
            # Lấy phần thực
            sb = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fft_masked)))
            subbands.append(sb)
        return subbands

    def inverse(self, subbands):
        # Cộng gộp tuyến tính
        return torch.stack(subbands, dim=0).sum(dim=0)