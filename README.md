# Tăng cường ảnh vân tay sử dụng Directional Filter Bank (DFB)

Dự án này mô phỏng bộ lọc định hướng (Directional Filter Bank - DFB) với 8 băng tần con và ứng dụng của nó trong việc tăng cường chất lượng ảnh vân tay. Mã nguồn được viết bằng Python sử dụng thư viện PyTorch để xử lý ảnh trong miền tần số (FFT).

## Các bài báo tham khảo
Dự án dựa trên các nghiên cứu khoa học sau:
1. **A Filter Bank for the Directional Decomposition of Images: Theory and Design** - Bamberger, R.H. and Smith, M.J.T.
2. **FINGERPRINT ENHANCEMENT BASED ON THE DIRECTIONAL FILTER BANK** - Park, Sang-il, Mark JT Smith, and Jun Jae Lee
3. **New Fingerprint Image Enhancement Using Directional Filter Bank** - Oh, Sang Keun

## Cài đặt
Cài đặt các thư viện cần thiết:

```bash
pip install torch numpy matplotlib pillow torchvision
```

## Cách chạy chương trình
1. Chuẩn bị ảnh đầu vào `fingerprint.jpg`.

2. Chạy file `main.py`:

```bash
python main.py
```

## Kết quả
Sau khi chương trình chạy xong, các file kết quả sẽ được lưu trong thư mục hiện tại:
- `enhanced_fingerprint.jpg`: Ảnh vân tay sau khi đã được xử lý tăng cường.
- `enhanced_fingerprint_binary.jpg`: Ảnh vân tay sau khi được nhị phân hóa (đen trắng).
- `subbands_vis.png`: Hình ảnh trực quan hóa 8 băng tần con (subbands) được tách ra từ bộ lọc DFB.
- `comparison_result.png`: Hình ảnh tổng hợp so sánh giữa ảnh gốc, ảnh tăng cường và kết quả nhị phân hóa.
