import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys

# Khắc phục lỗi in tiếng Việt (UnicodeEncodeError) trên console Windows
sys.stdout.reconfigure(encoding='utf-8')
def test_image_with_model(model_path, image_path, z):
    """
    Hàm kiểm tra dự đoán của mô hình với một hình ảnh mới.
    
    Tham số:
    - model_path: Đường dẫn đến mô hình được lưu trong máy tính (ví dụ: 'ann_model.pkl')
    - image_path: Đường dẫn đến tập tin hình ảnh cần dự đoán
    - z: Kích thước ảnh vuông (ảnh sẽ được resize về z x z)
    """
    # 1. Đọc mô hình vào bộ nhớ
    print(f"Đang tải mô hình từ: {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # 2. Đọc hình ảnh vào bộ nhớ
    print(f"Đang đọc hình ảnh từ: {image_path}...")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Không thể đọc được hình ảnh. Vui lòng kiểm tra lại đường dẫn: {image_path}")
        return

    # Để hiển thị đúng màu ảnh gốc bằng matplotlib, cần chuyển BGR (OpenCV) sang RGB
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 3. Chuyển ảnh màu thành ảnh xám
    print("Đang chuyển đổi ảnh màu sang ảnh xám...")
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 4. Chuyển ảnh xám thành ảnh vuông có kích thước z x z
    print(f"Đang thay đổi kích thước ảnh về {z}x{z} pixels...")
    resized_image = cv2.resize(gray_image, (z, z))

    # 5. Chuyển ảnh vuông thành vector và chuẩn hóa bằng MinMaxScaler
    print("Đang chuyển đổi thành vector và chuẩn hóa bằng MinMaxScaler...")
    # Flatten ảnh vuông z*z thành vector 1 chiều có số cột là z*z (shape: 1 dòng, z*z cột)
    vector_image = resized_image.flatten()
    
    # Để sử dụng MinMaxScaler chuẩn hóa các pixel theo giá trị từ (0 -> 1), 
    # ta reshape vector này theo dạng cột (N, 1), sau đó fit_transform và reshape trở lại (1, N)
    scaler = MinMaxScaler()
    vector_image_scaled = scaler.fit_transform(vector_image.reshape(-1, 1)).reshape(1, -1)

    # 6. Sử dụng vector này để mô hình dự đoán nhãn lớp
    print("Đang tiến hành dự đoán...")
    predicted_label = model.predict(vector_image_scaled)
    # Nếu mô hình lấy đầu ra là Cat/Dog dạng chuỗi hoặc 0/1 thì in trực tiếp
    label_result = str(predicted_label[0])
    print(f"KẾT QUẢ DỰ ĐOÁN: {label_result}")

    # 7. Hiển thị lên màn hình ảnh gốc, ảnh vuông, kết quả dự đoán
    print("Đang hiển thị biểu đồ...")
    plt.figure(figsize=(10, 5))

    # Hiển thị ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title("Ảnh Màu Gốc")
    plt.axis("off") # Tắt trục tọa độ

    # Hiển thị ảnh vuông (đã chuyển xám)
    plt.subplot(1, 2, 2)
    plt.imshow(resized_image, cmap='gray')
    plt.title(f"Ảnh Vuông {z}x{z} (Ảnh Xám)\nNhãn dự đoán: {label_result}", color='red', fontweight='bold')
    plt.axis("off")

    # Hiệu chỉnh khoảng cách và show
    plt.tight_layout()
    plt.show()

# ==========================================
# PHẦN CHẠY THỬ NGHIỆM (MAIN)
# ==========================================
if __name__ == "__main__":
    # Thay đổi các đường dẫn dưới đây cho phù hợp với máy tính của bạn
    MY_MODEL_PATH = 'ann_model.pkl'         # Đường dẫn đến mô hình đã lưu
    MY_IMAGE_PATH = r'D:\\HMUD-K22\\Image\\img1.jpg' # Thêm chữ 'r' phía trước để không bị lỗi dấu gạch chéo
    # Lưu ý: Nếu đường dẫn file ảnh của bạn có đuôi (ví dụ: .jpg, .png), hãy thêm vào cuối.
    # MY_IMAGE_PATH = r'D:\HMUD-K22\Image\img1.jpg'
    Z_SIZE = 64                             # Kích thước vuông z x z (đồng bộ với quá trình training trước đó)

    # Gọi hàm dự đoán
    test_image_with_model(MY_MODEL_PATH, MY_IMAGE_PATH, Z_SIZE)
