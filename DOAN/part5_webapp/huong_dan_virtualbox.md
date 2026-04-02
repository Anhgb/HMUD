# HƯỚNG DẪN TRIỂN KHAI VIRTUALBOX UBUNTU
## Đồ Án: Nhận Dạng Ngôn Ngữ Ký Hiệu — FastAPI Server trên VM

---

## BƯỚC 1: Cài Đặt VirtualBox & Tạo Máy Ảo Ubuntu

### 1.1 Tải và cài VirtualBox
- Tải VirtualBox: https://www.virtualbox.org/wiki/Downloads
- Tải Ubuntu Server 22.04 LTS ISO: https://ubuntu.com/download/server

### 1.2 Tạo máy ảo mới
```
- Name: SLR-API-Server
- Type: Linux / Ubuntu (64-bit)
- RAM: 4096 MB (4GB) — tối thiểu 2GB
- Disk: 20GB (VDI, Dynamic)
```

### 1.3 Cấu hình máy ảo TRƯỚC khi cài Ubuntu
Vào **Settings → Network:**
- **Adapter 1**: Chọn `Bridged Adapter` → chọn card mạng WiFi/Ethernet của máy thật
- (Bridged giúp VM có IP riêng trong cùng mạng LAN, máy host có thể kết nối dễ dàng)

### 1.4 Cài Ubuntu Server
- Boot từ ISO
- Chọn ngôn ngữ: English
- Bỏ qua cập nhật installer
- Cấu hình mạng: để DHCP tự động
- Đặt username: `slruser` | password tùy ý
- **Tích chọn: Install OpenSSH Server** ← quan trọng!
- Hoàn tất cài đặt, reboot

---

## BƯỚC 2: Cấu Hình Mạng

### 2.1 Tìm IP của máy ảo (sau khi Ubuntu boot)
```bash
ip addr show
# Hoặc
hostname -I
```
Ghi nhớ IP, ví dụ: `192.168.1.105`

### 2.2 Kết nối SSH từ máy Windows host (tùy chọn)
```powershell
# Trên máy Windows
ssh slruser@192.168.1.105
```

### 2.3 (THAY THẾ) Nếu dùng NAT thay vì Bridged — Port Forwarding
Vào **Settings → Network → Adapter 1 → Advanced → Port Forwarding:**

| Name | Protocol | Host IP | Host Port | Guest IP | Guest Port |
|------|----------|---------|-----------|----------|------------|
| SSH  | TCP      | 127.0.0.1 | 2222  | (blank)  | 22   |
| API  | TCP      | 127.0.0.1 | 8000  | (blank)  | 8000 |

Khi đó, từ máy host gọi API qua: `http://localhost:8000`

---

## BƯỚC 3: Cài Đặt Môi Trường Trên Ubuntu VM

```bash
# Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y

# Cài Python 3.10+ và pip
sudo apt install -y python3.10 python3.10-venv python3-pip git curl

# Kiểm tra phiên bản
python3 --version
pip3 --version
```

### 3.1 Upload code API lên VM

**Cách A: Dùng SCP (khuyên dùng)**
```powershell
# Từ máy Windows — copy thư mục part4_api lên VM
scp -r D:\HMUD-K22\DOAN\part4_api\   slruser@192.168.1.105:~/slr-api/
scp -r D:\HMUD-K22\DOAN\part3_huan_luyen\models\ slruser@192.168.1.105:~/slr-api/models/
scp    D:\HMUD-K22\DOAN\requirements.txt slruser@192.168.1.105:~/slr-api/
```

**Cách B: Dùng Git**
```bash
# Trên VM
git clone https://github.com/YOUR_USERNAME/slr-project.git ~/slr-api
```

**Cách C: Shared Folder VirtualBox**
```
Settings → Shared Folders → Add → chọn thư mục DOAN trên Windows
→ Folder Name: doan_shared
→ Tích: Auto-mount, Make Permanent
```
```bash
# Trên Ubuntu
sudo adduser slruser vboxsf
sudo mount -t vboxsf doan_shared /mnt/doan
```

### 3.2 Tạo Virtual Environment và cài thư viện

```bash
# SSH vào VM
ssh slruser@192.168.1.105

# Tạo thư mục dự án
mkdir -p ~/slr-api
cd ~/slr-api

# Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Cài thư viện (chỉ cần phần API)
pip install --upgrade pip
pip install fastapi uvicorn tensorflow-cpu numpy

# Kiểm tra TensorFlow
python -c "import tensorflow as tf; print('TF OK:', tf.__version__)"
```

---

## BƯỚC 4: Chạy FastAPI Server Trên VM

### 4.1 Cấu trúc thư mục trên VM
```
~/slr-api/
├── api_server.py           ← FastAPI server
├── models/
│   ├── best_model.h5       ← Model đã train
│   └── label_names.npy     ← Tên các class
└── .venv/
```

### 4.2 Chỉnh đường dẫn trong api_server.py
```python
# Mở api_server.py và sửa:
MODEL_PATH  = "/home/slruser/slr-api/models/best_model.h5"
LABEL_PATH  = "/home/slruser/slr-api/models/label_names.npy"
```

### 4.3 Chạy server
```bash
cd ~/slr-api
source .venv/bin/activate

# Chạy thủ công (dùng để test)
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Chạy với reload (khi dev)
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 4.4 Kiểm tra từ máy Windows
```powershell
# Test Health Check
curl http://192.168.1.105:8000/

# Test API Info
curl http://192.168.1.105:8000/info

# Mở Swagger UI trong trình duyệt
# http://192.168.1.105:8000/docs
```

---

## BƯỚC 5: Tự Động Khởi Động API (Systemd Service)

Tạo service để API tự chạy khi VM boot:

```bash
sudo nano /etc/systemd/system/slr-api.service
```

Nội dung file:
```ini
[Unit]
Description=Sign Language Recognition FastAPI Server
After=network.target

[Service]
User=slruser
WorkingDirectory=/home/slruser/slr-api
Environment="PATH=/home/slruser/slr-api/.venv/bin"
ExecStart=/home/slruser/slr-api/.venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
# Kích hoạt và khởi động service
sudo systemctl daemon-reload
sudo systemctl enable slr-api
sudo systemctl start slr-api

# Kiểm tra trạng thái
sudo systemctl status slr-api

# Xem log
journalctl -u slr-api -f
```

---

## BƯỚC 6: Chạy Web App Streamlit Trên Máy Host (Windows)

```powershell
# Trên máy Windows
cd D:\HMUD-K22\DOAN
pip install -r requirements.txt

# Chạy Streamlit Web App
streamlit run part5_webapp/app.py
```

Trong giao diện Streamlit:
1. Sửa **URL FastAPI Server** thành `http://192.168.1.105:8000`
2. Nhấn **Kiểm tra kết nối API** để xác nhận
3. Nhấn **▶️ Bắt Đầu** để mở webcam và nhận diện real-time!

---

## LUỒNG HỆ THỐNG HOÀN CHỈNH

```
[Máy Windows - Host]                    [VirtualBox Ubuntu VM]
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  Browser / Streamlit App     │        │  FastAPI Server              │
│                              │        │  (uvicorn :8000)             │
│  Webcam → OpenCV             │        │                              │
│  → MediaPipe Landmarks       │        │  POST /predict               │
│  → Buffer 60 frames   ──────►│──LAN──►│  → Load model.h5            │
│  ← Nhận kết quả JSON  ◄──────│◄───────│  → LSTM predict             │
│  → Hiển thị lên video        │        │  → Trả về JSON              │
└──────────────────────────────┘        └──────────────────────────────┘
     IP: 192.168.1.100                       IP: 192.168.1.105:8000
```

---

## CHECKLIST HOÀN THÀNH DỰ ÁN

### Phần 1 - Thu thập dữ liệu
- [ ] Quay video 30-50 clips/class cho 10 class
- [ ] Gán nhãn trên Label Studio (hoặc dùng thu_thap_truc_tiep.py)
- [ ] Export JSON từ Label Studio

### Phần 2 - Tiền xử lý
- [ ] Chạy `extract_features.py` → tạo X_sequences.npy, y_labels.npy
- [ ] Kiểm tra `python extract_features.py --mode check`

### Phần 3 - Huấn luyện mô hình
- [ ] Chạy `train_model.py`
- [ ] Kiểm tra biểu đồ trong `part3_huan_luyen/bieu_do/`
- [ ] Accuracy > 80% là tốt

### Phần 4 - FastAPI
- [ ] Copy model lên VirtualBox Ubuntu
- [ ] Chạy uvicorn thành công
- [ ] Test Swagger UI: http://VM_IP:8000/docs

### Phần 5 - Web App
- [ ] Chạy `streamlit run part5_webapp/app.py`
- [ ] Nhận diện real-time hoạt động
- [ ] Systemd service tự khởi động
