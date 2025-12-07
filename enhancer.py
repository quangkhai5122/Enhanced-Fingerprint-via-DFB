import torch
import torch.nn.functional as F

class FingerprintEnhancer:
    def __init__(self, dfb, block_size=16):
        """
        block_size: Kích thước block để tính năng lượng cục bộ (local energy).
        Bài báo Oh et al. dùng block để ước lượng hướng.
        """
        self.dfb = dfb
        self.block_size = block_size
        
    def compute_energy(self, subbands):
        energies = []
        # Subbands bây giờ có cùng kích thước với ảnh gốc.
        # Ta dùng AvgPool để tính tổng năng lượng trong cửa sổ trượt.
        
        for subband in subbands:
            # L1 Energy: Sum of absolute values
            abs_sb = torch.abs(subband)
            
            # Tính trung bình cục bộ (đại diện cho năng lượng)
            # Dùng Kernel bằng block_size, stride=1 để có energy map mượt (dense prediction)
            # Hoặc stride=block_size để tính theo block rời rạc.
            # Để ảnh mượt mà không bị artifact khối, ta dùng stride=1 (Dense map) và padding.
            
            # Tuy nhiên, bài báo nói "blocks", ta dùng stride nhỏ (ví dụ 4) để trade-off speed/quality
            # Hoặc giữ nguyên logic block rời rạc.
            
            pad = self.block_size // 2
            # E_k shape sẽ bằng shape ảnh gốc (nhờ padding=same)
            E_k = F.avg_pool2d(
                abs_sb, 
                kernel_size=self.block_size, 
                stride=1, 
                padding=pad,
                divisor_override=1
            )
            
            # Do padding có thể làm lệch size 1 chút, crop về size chuẩn
            if E_k.shape != subband.shape:
                E_k = E_k[:, :, :subband.shape[2], :subband.shape[3]]
            
            energies.append(E_k)
            
        return energies

    def enhance(self, image):
        # 1. Analysis (FFT based)
        # subbands: List of [B, 1, H, W] tensors
        subbands = self.dfb(image)
        
        # 2. Energy Estimation
        energies = self.compute_energy(subbands)
        E_stack = torch.stack(energies, dim=1) # (B, 8, 1, H, W)
        
        # 3. Directional Selection & Weighting
        # Eq. 5: w_k = E_k / E_max
        E_max, _ = torch.max(E_stack, dim=1, keepdim=True)
        E_max = E_max + 1e-8
        
        weights = E_stack / E_max
        
        # Eq. 6: Normalization (Energy Preservation)
        # Scale weights sao cho tổng năng lượng sau khi weight bằng tổng năng lượng gốc
        total_E_in = torch.sum(E_stack, dim=1, keepdim=True)
        total_E_out = torch.sum(weights * E_stack, dim=1, keepdim=True)
        
        norm_factor = total_E_in / (total_E_out + 1e-8)
        final_weights = weights * norm_factor
        
        # 4. Enhancement application
        enhanced_subbands = []
        for k, subband in enumerate(subbands):
            # subband: (B, 1, H, W)
            # w: (B, 1, 1, H, W) -> squeeze dim 2 -> (B, 1, H, W)
            w = final_weights[:, k, ...].squeeze(1)
            enhanced_subbands.append(subband * w)
            
        # 5. Synthesis
        enhanced_image = self.dfb.inverse(enhanced_subbands)
        
        return enhanced_image, subbands