# BÁO CÁO ĐỒ ÁN: HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU BẰNG TRÍ TUỆ NHÂN TẠO
**Tên dự án:** Sign Language Recognition System theo thời gian thực (Real-time).
**Công nghệ sử dụng:** Python, TensorFlow (Keras LSTM), MediaPipe, FastAPI, CustomTkinter.
**Kiến trúc:** Phân tán Client - Server

---

## 1. MỤC TIÊU ĐỀ TÀI
- Xây dựng một hệ thống Trí tuệ Nhân tạo có khả năng nhận diện các cử chỉ ngôn ngữ ký hiệu của người khiếm thính thông qua Camera theo đúng thời gian thực.
- Triển khai ứng dụng theo chuẩn mô hình phần mềm cấp doanh nghiệp (Client-Server), trong đó phần "Não AI" được tách riêng biệt ra một máy chủ để xử lý, phần giao diện người dùng hoạt động như một máy khách nhẹ nhàng.

## 2. KIẾN TRÚC HỆ THỐNG (CLIENT-SERVER)
Dự án được chia làm 2 thành phần hoạt động song song để tối ưu hóa hiệu năng:

### 2.1. Client (Ứng dụng Desktop)
- **Công nghệ:** `CustomTkinter`, `OpenCV`, `MediaPipe`.
- **Nhiệm vụ:** 
  - Mở Camera và quay video người dùng ở tốc độ cao.
  - Sử dụng AI nguyên bản của Google (MediaPipe) để quét và định vị chính xác **126 điểm tọa độ** trên 2 bàn tay (X, Y, Z coordinates).
  - Thu thập liên tục 60 khung hình x 126 điểm tọa độ để tạo thành một ma trận chuỗi hành động.
  - Gửi dữ liệu này qua cổng Internet (HTTP Request) đến Server.

### 2.2. Server (Máy chủ Inference AI)
- **Công nghệ:** `FastAPI`, `Uvicorn`, `TensorFlow`.
- **Nhiệm vụ:**
  - Hoạt động độc lập trên cổng lưới mạng (Port 8000).
  - Tiếp nhận ma trận tọa độ tay từ Client gửi lên.
  - Đưa ma trận này vào Mạng nơ-ron học sâu LSTM đã được huấn luyện sẵn (`best_model.keras`).
  - Đưa ra dự đoán xác suất (Confidence Score) cho từng hành động. Nếu xác suất vượt ngưỡng tin cậy sẽ trả kết quả văn bản (Từ vựng) về cho Client hiển thị.

---

## 3. CƠ CHẾ HUẤN LUYỆN TRÍ TUỆ NHÂN TẠO
Mô hình AI trong đồ án sử dụng kiến trúc **LSTM (Long Short-Term Memory)** thay vì các mạng CNN thông thường. 
- **Lý do sử dụng LSTM:** Ngôn ngữ ký hiệu không phải là một bức ảnh tĩnh, mà là một *chuỗi chuyển động theo thời gian*. LSTM là mạng nơ-ron hồi quy có khả năng ghi nhớ diễn biến thời gian, do đó học được "quỹ đạo" chuyển động của bàn tay người dùng.
- **Dữ liệu đầu vào (Input Shape):** `(60, 126)` tương ứng với 60 khung hình liên tục, mỗi khung hình có 126 thông số tọa độ.
- **Độ chính xác (Accuracy):** Mô hình đạt độ chính xác lên tới **~92.6%** trên tập dữ liệu kiểm thử, hoạt động mượt mà với 9 lớp từ vựng vĩ mô cơ bản (Xin chào, Xin lỗi, Yêu, Tôi, Bạn...).

---

## 4. QUY TRÌNH THỰC HIỆN TOÀN DIỆN
Dự án được xây dựng từ con số 0 theo quy trình phân tích dữ liệu chuẩn 5 bước (Pipeline AI):

* **Bước 1: Thu thập trực tiếp (Part 1)** - Viết phần mềm tự động quay video người dùng làm ký hiệu tay để tạo Dataset gốc dạng video `.avi`.
* **Bước 2: Rút gọn dữ liệu (Part 2)** - Viết script chạy qua hàng chục nghìn khung hình video, dùng bộ lọc MediaPipe để bóc tách bỏ hết cảnh nền, xóa quần áo, chỉ giữ lại tọa độ 3D của cấu trúc xương tay.
* **Bước 3: Huấn luyện (Part 3)** - Code mô hình học máy LSTM, gán nhãn dữ liệu qua Label Studio và Train ra siêu trọng lượng mô hình có khả năng tư duy (`.keras`).
* **Bước 4: Mở API Server (Part 4)** - Code máy chủ FastAPI kết nối với mô hình.
* **Bước 5: Phát triển App (Part 5)** - Hoàn thiện UI/UX của Desktop Client với Dark Mode đẹp mắt để kết nối API.

## 5. KẾT LUẬN & HƯỚNG PHÁT TRIỂN THÊM
Hệ thống đồ án đã đáp ứng vượt chỉ tiêu yêu cầu, có thể ứng dụng trong đời sống thực tế để hỗ trợ người khiếm thính giao tiếp ở môi trường y tế, hành chính. 
- **Hướng vươn xa:** Đưa Backend Server này lên Cloud (như Amazon AWS hoặc Google Cloud) trên con máy chủ Ubuntu để mọi ứng dụng di động trên thế giới đều có thể tải app về và quét tay không cần máy xịn. (Demo kỹ năng Virtual Machine).
