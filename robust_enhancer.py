import torch
import torch.nn.functional as F
import numpy as np

class RobustFingerprintEnhancer:
    def __init__(self, dfb, denoise_strength=1.0, block_size=16):
        self.dfb = dfb
        self.denoise_strength = denoise_strength # Lambda factor
        self.block_size = block_size

    def normalize_image(self, img):
        mean = torch.mean(img)
        std = torch.std(img)
        norm_img = (img - mean) / (std + 1e-8)
        return norm_img

    def estimate_noise_sigma(self, x):
        """
        Ước lượng độ lệch chuẩn của nhiễu bằng Median Absolute Deviation (MAD).
        Robust hơn std dev thông thường.
        sigma = median(|x|) / 0.6745 (cho phân phối chuẩn)
        """
        return torch.median(torch.abs(x)) / 0.6745

    def soft_threshold(self, x, threshold):
        """
        Donoho's Soft Thresholding:
        Làm sạch nền, giữ lại các đặc trưng mạnh (ridges).
        """
        return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0, device=x.device))

    def compute_directional_energy(self, subbands):
        """Tính bản đồ năng lượng local cho mỗi hướng"""
        energies = []
        kernel_size = self.block_size * 2 
        pad = kernel_size // 2
        for sb in subbands:
            sq_sb = sb ** 2
            E = F.avg_pool2d(sq_sb, kernel_size=kernel_size, stride=1, padding=pad)
            if E.shape != sb.shape:
                E = E[:, :, :sb.shape[2], :sb.shape[3]]
            energies.append(E)
        return torch.stack(energies, dim=1) # (B, 8, 1, H, W)

    def enhance(self, image):
        image = self.normalize_image(image)
        subbands = self.dfb(image)
        enhanced_subbands = []
        
        # Tính toán năng lượng định hướng toàn cục để tìm weighting
        energies = self.compute_directional_energy(subbands)
        max_energy, _ = torch.max(energies, dim=1, keepdim=True)

        directional_weights = energies / (max_energy + 1e-8)
        directional_weights = directional_weights ** 2 

        # Xử lý từng Subband
        for k, sb in enumerate(subbands):
            # 1. Denoising (Soft Thresholding)
            sigma = self.estimate_noise_sigma(sb)
            threshold = sigma * self.denoise_strength
            
            sb_denoised = self.soft_threshold(sb, threshold)
            
            # 2. Boosting (Nhân trọng số định hướng)
            # w: (B, 1, 1, H, W) -> (B, 1, H, W)
            w = directional_weights[:, k, ...].squeeze(1)
            
            # Kết hợp: Subband sạch * Trọng số hướng
            sb_enhanced = sb_denoised * w * 2.0 # Nhân 2.0 để bù lại năng lượng bị mất do threshold
            
            enhanced_subbands.append(sb_enhanced)
            
        # Reconstruction
        enhanced_image = self.dfb.inverse(enhanced_subbands)
        
        return enhanced_image, enhanced_subbands