import torch
import torch.nn as nn
import torch.fft
import numpy as np

class DirectionalFilterBank(nn.Module):
    def __init__(self, num_bands=8, device='cpu'):
        super(DirectionalFilterBank, self).__init__()
        self.num_bands = num_bands
        self.device = device
        
    def generate_angular_masks(self, shape):
        H, W = shape
        # Tạo lưới toạ độ tần số (Frequency Grid); Shifted: Gốc (0,0) nằm giữa ảnh
        y = torch.arange(H, device=self.device) - H // 2
        x = torch.arange(W, device=self.device) - W // 2
        Y, X = torch.meshgrid(y, x, indexing='ij')

        angles = torch.atan2(Y, X)
        masks = []
        step = np.pi / self.num_bands # pi / 8
        
        for k in range(self.num_bands):
            start_angle = k * step - np.pi/2 # Bắt đầu từ trục dọc
            end_angle = (k + 1) * step - np.pi/2

            center_angle = (start_angle + end_angle) / 2

            angles_mod = angles % np.pi
            start_mod = start_angle % np.pi
            end_mod = end_angle % np.pi
            
            if start_mod < end_mod:
                mask = (angles_mod >= start_mod) & (angles_mod < end_mod)
            else: # Vượt qua mốc pi
                mask = (angles_mod >= start_mod) | (angles_mod < end_mod)
                
            masks.append(mask.float())
            
        return torch.stack(masks)

    def forward(self, x):
        """
        Analysis: Ảnh -> 8 Subbands cùng kích thước
        """
        B, C, H, W = x.shape
        
        # 1. FFT: Dùng fft2 đầy đủ để dễ xử lý mask đối xứng.
        fft_x = torch.fft.fftshift(torch.fft.fft2(x))
        
        # 2. Generate Masks (Lazy init)
        if not hasattr(self, 'masks') or self.masks.shape[-2:] != (H, W):
            self.masks = self.generate_angular_masks((H, W))
            
        # 3. Apply Masks
        subbands = []
        for k in range(self.num_bands):
            mask = self.masks[k].unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            fft_masked = fft_x * mask
            
            # 4. IFFT
            sb = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fft_masked)))
            subbands.append(sb)
            
        return subbands

    def inverse(self, subbands):
        return torch.stack(subbands, dim=0).sum(dim=0)