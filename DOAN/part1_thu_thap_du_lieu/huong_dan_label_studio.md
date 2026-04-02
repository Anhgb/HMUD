# HƯỚNG DẪN THU THẬP DỮ LIỆU & GÁN NHÃN VỚI LABEL STUDIO
## Đồ Án: Nhận Dạng Ngôn Ngữ Ký Hiệu (Sign Language Recognition)

---

## 1. TỪ VỰNG GỢI Ý (10 Classes)

Nên chọn **10 từ/ký hiệu cơ bản** để vừa đủ phong phú cho demo, vừa khả thi về thời gian:

| STT | Từ Vựng | Ký Hiệu Mô Tả |
|-----|---------|----------------|
| 0   | Xin chào | Vẫy tay |
| 1   | Cảm ơn   | Tay áp vào ngực, cúi đầu nhẹ |
| 2   | Tôi      | Trỏ ngón vào bản thân |
| 3   | Bạn      | Trỏ ngón về phía trước |
| 4   | Yêu      | Khoanh tay hình tim |
| 5   | Không    | Lắc ngón trỏ |
| 6   | Có       | Gật đầu + ngón cái giơ lên |
| 7   | Giúp đỡ  | Hai tay nắm lại giơ lên |
| 8   | Xin lỗi  | Nắm tay xoay tròn ở ngực |
| 9   | Tạm biệt | Vẫy tay từ biệt |

---

## 2. HƯỚNG DẪN QUAY VIDEO

### Thiết lập môi trường quay:
- **Ánh sáng**: Dùng đèn đặt PHÍA TRƯỚC mặt (không dùng đèn sau lưng gây bóng ngược)
- **Nền**: Dùng tường trắng hoặc rèm có màu đơn sắc, tương phản với màu da
- **Góc máy**: Camera ngang tầm ngực, nhìn thẳng vào người quay (không nghiêng)
- **Khoảng cách**: Cách camera 0.5 - 1.5 mét, toàn bộ 2 tay trong khung hình

### Thông số video:
- **Độ phân giải**: 640x480 hoặc 1280x720
- **FPS**: 30fps
- **Thời lượng mỗi clip**: 2-3 giây/lần ký hiệu
- **Số video/class**: Quay **ít nhất 30-50 video** mỗi từ
- **Đa dạng**: Quay với nhiều người khác nhau, ánh sáng khác nhau

### Lưu ý quan trọng:
- Mặc áo sẫm màu để tay nổi bật
- Giữ tay trong vùng sáng
- Tránh vật cản che khuất tay

---

## 3. SETUP PROJECT TRÊN LABEL STUDIO

### Bước 1: Cài đặt Label Studio
```bash
pip install label-studio
label-studio start
# Truy cập: http://localhost:8080
```

### Bước 2: Tạo Project mới
1. Click **"Create Project"**
2. **Project Name**: `Sign_Language_Recognition`
3. Chọn tab **"Labeling Setup"**
4. Chọn template: **"Video Classification"** hoặc **"Custom Template"**

### Bước 3: Cấu hình Labeling Template
Dán XML sau vào phần Custom Template:

```xml
<View>
  <Video name="video" value="$video" framerate="30"/>
  <VideoRectangle name="box" toName="video"/>
  <Labels name="label" toName="video">
    <Label value="xin_chao" background="#FF6B6B"/>
    <Label value="cam_on" background="#4ECDC4"/>
    <Label value="toi" background="#45B7D1"/>
    <Label value="ban" background="#96CEB4"/>
    <Label value="yeu" background="#FFEAA7"/>
    <Label value="khong" background="#DDA0DD"/>
    <Label value="co" background="#98D8C8"/>
    <Label value="giup_do" background="#F7DC6F"/>
    <Label value="xin_loi" background="#BB8FCE"/>
    <Label value="tam_biet" background="#85C1E9"/>
  </Labels>
</View>
```

**HOẶC** nếu gán nhãn đơn giản theo clip (khuyên dùng):
```xml
<View>
  <Video name="video" value="$video"/>
  <Choices name="gesture" toName="video" choice="single">
    <Choice value="xin_chao"/>
    <Choice value="cam_on"/>
    <Choice value="toi"/>
    <Choice value="ban"/>
    <Choice value="yeu"/>
    <Choice value="khong"/>
    <Choice value="co"/>
    <Choice value="giup_do"/>
    <Choice value="xin_loi"/>
    <Choice value="tam_biet"/>
  </Choices>
</View>
```

### Bước 4: Import video vào Label Studio
1. Click **"Import"** trong project
2. Upload từng video hoặc dùng **Local Storage** / **URL**
3. Nên đặt tên video theo quy ước: `xin_chao_001.mp4`, `cam_on_002.mp4`...

### Bước 5: Gán nhãn
1. Click vào từng task (video)
2. Xem video, chọn nhãn tương ứng
3. Click **"Submit"** để lưu

---

## 4. EXPORT DỮ LIỆU TỪ LABEL STUDIO

### Format xuất khuyên dùng: **JSON**
1. Click **"Export"** trong project
2. Chọn format **"JSON"**
3. Lưu file `annotations.json`

### Cấu trúc JSON xuất ra:
```json
[
  {
    "id": 1,
    "data": {
      "video": "/path/to/xin_chao_001.mp4"
    },
    "annotations": [
      {
        "result": [
          {
            "value": {
              "choices": ["xin_chao"]
            },
            "type": "choices"
          }
        ]
      }
    ]
  }
]
```

### Script đọc JSON từ Label Studio (xem file parse_label_studio.py)

---

## 5. CẤU TRÚC THƯ MỤC DỰ ÁN

```
DOAN/
├── part1_thu_thap_du_lieu/
│   ├── huong_dan_label_studio.md   # File này
│   └── raw_videos/
│       ├── xin_chao/
│       │   ├── xin_chao_001.mp4
│       │   └── ...
│       ├── cam_on/
│       └── ...
├── part2_trich_xuat_dac_trung/
│   ├── extract_features.py         # Script trích xuất landmark
│   ├── parse_label_studio.py       # Parse JSON từ Label Studio
│   └── dataset/
│       ├── X_sequences.npy
│       └── y_labels.npy
├── part3_huan_luyen/
│   ├── train_model.py
│   └── models/
├── part4_api/
│   └── api_server.py
├── part5_webapp/
│   └── app.py (Streamlit)
└── requirements.txt
```
