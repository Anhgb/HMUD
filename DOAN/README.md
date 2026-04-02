# HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU (Sign Language Recognition)

Đây là Đồ án kết thúc học phần, sử dụng Trí tuệ Nhân tạo (Mô hình học sâu LSTM) để nhận diện và dịch trực tiếp Ngôn ngữ Ký hiệu bằng Camera. Hệ thống được xây dựng theo kiến trúc Client-Server.

## 📸 Hình ảnh Demo Hệ thống

*(Kéo thả ảnh demo giao diện, ảnh chụp lúc hệ thống nhận diện đúng ký hiệu vào phía dưới dòng này)*:
<!-- Kéo thả ảnh vào đây. Github sẽ tự đổi nó thành link ảnh -->



## 📄 Tài liệu Báo Cáo Đồ Án

*(Kéo thả file báo cáo Word/PDF và Slide vào phía dưới dòng này)*:
<!-- Kéo thả file tài liệu vào đây -->



---
## Cấu trúc Mã Nguồn

Dự án tuân theo quy trình AI thực tế chia làm 5 phần và đáp ứng chuẩn thiết kế API:

1. **`part1_thu_thap_du_lieu/`**: Thu thập chuỗi khung hình video qua Camera.
2. **`part2_trich_xuat_dac_trung/`**: Trích xuất dữ liệu xương bàn tay (126 features) nhờ MediaPipe. 
3. **`part3_huan_luyen/`**: Logic huấn luyện học máy với mạng LTSM (Long Short-Term Memory).
4. **`part4_api/`** (Backend Server): Đóng gói mô hình AI thành API Server RESTful bằng FastAPI.
5. **`part5_webapp/`** (Frontend App): Giao diện Desktop Client gọi API Server thông qua CustomTkinter.

## Cách chạy dự án
Chỉ cần mở file `khoi_dong_he_thong.bat`, hệ thống sẽ tự động bật 2 luồng Server và Client.
