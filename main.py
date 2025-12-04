import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from dfb import DirectionalFilterBank
from enhancer import FingerprintEnhancer
import os

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

def save_image(tensor, path, binarize=False):
    """
    Lưu ảnh với xử lý hậu kỳ để loại bỏ màu xám:
    1. Robust Normalization: Kéo dãn histogram, loại bỏ outlier.
    2. Binarization (Optional): Chuyển về đen trắng tuyệt đối (như bài báo).
    """
    img = tensor.squeeze().cpu().detach().numpy()
    
    # Robust Normalization (Khử màu xám) 
    # Thay vì lấy min/max tuyệt đối (dễ bị nhiễu hạt muối tiêu làm hỏng), ta lấy phân vị thứ 1 và 99 để bỏ qua nhiễu biên.
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    
    # Xử lý Gamma (Tăng tương phản trung gian), giúp làm tối phần đen và sáng phần trắng hơn
    img = img ** 2.0  # Gamma > 1 làm ảnh đậm hơn, ridges rõ hơn
    
    # Binarization (Theo paper cua Oh et al. )
    if binarize:
        # Ngưỡng đơn giản hoặc dùng Otsu (thư viện ngoài
        img = (img > 0.35).astype(float)
    
    plt.imsave(path, img, cmap='gray')
    print(f"Saved {path}")

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
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img = load_image('fingerprint.jpg', device)
    if img is None:
        print("Image not found, creating dummy.")
        img = torch.rand(1, 1, 256, 256).to(device)

    # Sử dụng DFB FFT
    dfb = DirectionalFilterBank(num_bands=8, device=device)
    enhancer = FingerprintEnhancer(dfb, block_size=16)
    
    print("Running enhancement...")
    enhanced_img, subbands = enhancer.enhance(img)
    
    visualize_subbands(subbands, 'subbands_vis.png')
    save_image(enhanced_img, 'enhanced_fingerprint.jpg', binarize=False)
    save_image(enhanced_img, 'enhanced_fingerprint_binary.jpg', binarize=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6)) 

    img_np = img.squeeze().cpu().detach().numpy()
    ax[0].imshow(img_np, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    res_np = enhanced_img.squeeze().cpu().detach().numpy()
    p1, p99 = np.percentile(res_np, 1), np.percentile(res_np, 99)
    res_np = np.clip((res_np - p1)/(p99 - p1 + 1e-8), 0, 1)
    
    ax[1].imshow(res_np, cmap='gray')
    ax[1].set_title('Enhanced')
    ax[1].axis('off')

    bin_np = (res_np > 0.57).astype(float)
    ax[2].imshow(bin_np, cmap='gray')
    ax[2].set_title('Binarized Result')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_result.png')
    print("Saved comparison_result.png")

if __name__ == '__main__':
    main()