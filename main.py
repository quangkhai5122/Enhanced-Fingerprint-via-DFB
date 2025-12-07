import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from soft_dfb import DirectionalFilterBank
from enhancer import FingerprintEnhancer
from robust_enhancer import RobustFingerprintEnhancer
import os
import cv2

def load_image(path, device='cpu'):
    if not os.path.exists(path): return None
    img = Image.open(path).convert('L')
    # Resize về kích thước chẵn để FFT chạy nhanh và chuẩn
    w, h = img.size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    img = img.resize((new_w, new_h))
    
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0).to(device)

def save_image_adaptive(tensor, path):
    """
    Hàm lưu ảnh thông minh:
    1. Robust Normalization cho ảnh Gray.
    2. Adaptive Binarization cho ảnh Binary (Khắc phục đứt đoạn).
    """
    img = tensor.squeeze().cpu().detach().numpy()
    
    # --- 1. Robust Normalize (cho ảnh xám) ---
    p5 = np.percentile(img, 5)   # Lấy phân vị 5% (tránh nhiễu đen)
    p95 = np.percentile(img, 95) # Lấy phân vị 95% (tránh nhiễu trắng)
    img_norm = np.clip((img - p5) / (p95 - p5 + 1e-8), 0, 1)
    
    # Lưu ảnh Enhanced Gray
    plt.imsave(path.replace('.jpg', '_gray.jpg'), img_norm, cmap='gray')
    
    # --- 2. Adaptive Binarization (QUAN TRỌNG) ---
    # Chuyển về 8-bit integer [0, 255]
    img_uint8 = (img_norm * 255).astype(np.uint8)
    
    # Dùng Gaussian Adaptive Thresholding
    # Block Size: 15 (hoặc 31 tùy độ phân giải), C: hằng số trừ đi (thường là 2-10)
    # Kỹ thuật này tính ngưỡng riêng cho từng vùng nhỏ, giúp nối liền vân tay ngay cả khi bị mờ.
    binarized = cv2.adaptiveThreshold(
        img_uint8, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, # Nền trắng, vân đen -> Dùng THRESH_BINARY. Nếu muốn nền đen vân trắng -> THRESH_BINARY_INV
        blockSize=15, 
        C=5
    )
    
    # Bài báo thường show nền trắng, vân đen.
    # Nếu kết quả ra nền đen, ta đảo ngược lại:
    # binarized = 255 - binarized
    
    cv2.imwrite(path.replace('.jpg', '_binary.jpg'), binarized)
    print(f"Saved {path}")
    return img_norm, binarized

def visualize_subbands(subbands, path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(subbands):
            sb = subbands[i].squeeze().cpu().detach().numpy()
            # Normalize từng band để nhìn rõ cấu trúc
            sb = (sb - sb.min()) / (sb.max() - sb.min() + 1e-8)
            ax.imshow(sb, cmap='gray')
            ax.set_title(f'Band {i}')
            ax.axis('off')
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    img = load_image('fingerprint.jpg', device)
    if img is None:
        print("Image not found, creating dummy.")
        img = torch.rand(1, 1, 256, 256).to(device)

    # Sử dụng DFB FFT
    dfb = DirectionalFilterBank(num_bands=8, device=device)
    enhancer = RobustFingerprintEnhancer(dfb, denoise_strength=0.75, block_size=16)
    # enhancer = FingerprintEnhancer(dfb, block_size=16)
    enhanced_img, subbands = enhancer.enhance(img)
    
    visualize_subbands(subbands, 'result_robustenhancer/subbands_vis.png')
    gray_res, bin_res = save_image_adaptive(enhanced_img, 'result_robustenhancer/final_result.jpg')
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img.squeeze().cpu().detach().numpy(), cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    ax[1].imshow(gray_res, cmap='gray')
    ax[1].set_title('Enhanced (Normalized)')
    ax[1].axis('off')
    
    ax[2].imshow(bin_res, cmap='gray')
    ax[2].set_title('Adaptive Binary')
    ax[2].axis('off')
    
    plt.savefig('result_robustenhancer/comparison_final.png')

if __name__ == '__main__':
    main()