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
        """
        Tạo các mặt nạ hình nêm (Wedge masks) trong miền tần số.
        Tương ứng với Ideal DFB partition (Hình 1 trong Bamberger 1992).
        """
        H, W = shape
        # Tạo lưới toạ độ tần số (Frequency Grid), Shifted: Gốc (0,0) nằm giữa ảnh
        y = torch.arange(H, device=self.device) - H // 2
        x = torch.arange(W, device=self.device) - W // 2
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        # Tính góc (Angle) của từng điểm tần số: [-pi, pi]
        angles = torch.atan2(Y, X)
        
        masks = []
        # Chia [-pi, pi] thành 8 phần. Mỗi phần rộng pi/4 (vì DFB đối xứng qua gốc)
        step = np.pi / self.num_bands # pi / 8
        
        for k in range(self.num_bands):
            start_angle = k * step - np.pi/2 
            end_angle = (k + 1) * step - np.pi/2
            
            # Tạo mask
            # Điểm (u,v) thuộc mask k nếu góc của nó nằm trong [start, end]
            # HOẶC nằm trong [start + pi, end + pi] 
            
            # Xử lý wrap-around pha pi: dùng cos để đo khoảng cách góc
            center_angle = (start_angle + end_angle) / 2
            
            # Khoảng cách góc: cos(a - b) càng gần 1 thì càng gần nhau. 
            # Dùng sigmoid để làm mềm biên (giảm ringing artifact)
            # hoặc dùng Hard threshold (như Ideal DFB). 
            
            # Chuẩn hóa angle về range [0, pi] để so sánh (do đối xứng)
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
        
        # FFT
        # x là thực, dùng rfft2 hoặc fft2. Dùng fft2 đầy đủ để dễ xử lý mask đối xứng.
        fft_x = torch.fft.fftshift(torch.fft.fft2(x))
        
        # Generate Masks (Lazy init)
        if not hasattr(self, 'masks') or self.masks.shape[-2:] != (H, W):
            self.masks = self.generate_angular_masks((H, W))
            
        # Apply Masks
        subbands = []
        for k in range(self.num_bands):
            mask = self.masks[k].unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            fft_masked = fft_x * mask
            
            # IFFT
            # Lấy phần thực vì ảnh vân tay là số thực
            sb = torch.real(torch.fft.ifft2(torch.fft.ifftshift(fft_masked)))
            subbands.append(sb)
            
        return subbands

    def inverse(self, subbands):
        """
        Synthesis: Cộng tất cả subbands lại.
        Vì sum(masks) = 1 (toàn bộ mặt phẳng tần số), nên sum(subbands) = original.
        """
        return torch.stack(subbands, dim=0).sum(dim=0)